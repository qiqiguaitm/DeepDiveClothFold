#!/usr/bin/python3
"""
ROS2 Policy Inference Node — 将 pi0.5 推理直接集成为 ROS2 节点

三种推理模式 (通过 --mode 参数选择):
  1. "ros2"      — 纯 ROS2 模式: JAX 推理在本节点内完成, 无 WebSocket
  2. "websocket" — 原版模式: 通过 WebSocket 连接外部 serve_policy.py
  3. "both"      — 同时启动: 本节点加载模型, 同时兼容 WebSocket 客户端

用法:
  # 模式 1: 纯 ROS2 (推荐, 最低延迟)
  ros2 run piper policy_inference_node.py --ros-args \
    -p mode:=ros2 \
    -p config_name:=pi05_flatten_fold_normal \
    -p checkpoint_dir:=gs://openpi-assets/checkpoints/pi05_base/params

  # 模式 2: WebSocket 客户端 (兼容旧的 serve_policy.py)
  ros2 run piper policy_inference_node.py --ros-args \
    -p mode:=websocket -p host:=localhost -p port:=8000

  # 模式 3: 两者兼有
  ros2 run piper policy_inference_node.py --ros-args \
    -p mode:=both \
    -p config_name:=pi05_flatten_fold_normal \
    -p checkpoint_dir:=gs://openpi-assets/checkpoints/pi05_base/params \
    -p ws_port:=8000

订阅:
  /camera_f/camera/color/image_raw  (sensor_msgs/Image)     — 头顶相机
  /camera_l/camera/color/image_raw  (sensor_msgs/Image)     — 左腕相机
  /camera_r/camera/color/image_raw  (sensor_msgs/Image)     — 右腕相机
  /puppet/joint_left            (sensor_msgs/JointState) — 左臂关节状态
  /puppet/joint_right           (sensor_msgs/JointState) — 右臂关节状态

发布:
  /policy/actions               (sensor_msgs/JointState) — 推理输出动作 (14 维)
  /master/joint_left            (sensor_msgs/JointState) — 左臂控制命令
  /master/joint_right           (sensor_msgs/JointState) — 右臂控制命令
"""

import json
import os
import sys

# ── 自动 re-exec: 确保在 kai0 venv 中运行 ────────────────────────
# ros2 run 通过 shebang (#!/usr/bin/env python3) 启动, 可能命中 conda 的
# python3.13 或系统 python3.12, 而本节点依赖 venv 中的 JAX/numpy/cv2 等包.
# 检测当前是否在 kai0 venv 中, 如果不是则 re-exec.
# KAI0_ROOT 查找顺序: 环境变量 > 相对路径推导 (source 和 install 两种布局)
_KAI0_ROOT = os.environ.get('KAI0_ROOT', '')
if not _KAI0_ROOT or not os.path.isdir(_KAI0_ROOT):
    # 从 __file__ 位置推导: source 布局 (ros2_ws/src/piper/scripts/ → ../../.. → kai0)
    for levels in [
        ('..', '..', '..', '..', 'kai0'),          # source: ros2_ws/src/piper/scripts/
        ('..', '..', '..', '..', '..', 'kai0'),     # install: ros2_ws/install/piper/lib/piper/
    ]:
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), *levels))
        if os.path.isdir(os.path.join(candidate, 'src', 'openpi')):
            _KAI0_ROOT = candidate
            break
    if not _KAI0_ROOT:
        # 最终回退: home-relative paths (avoid machine-specific hardcoded paths)
        for fallback in [os.path.expanduser('~/workspace/deepdive_kai0/kai0'),
                         os.path.join(os.path.expanduser('~'), 'kai0')]:
            if os.path.isdir(os.path.join(fallback, 'src', 'openpi')):
                _KAI0_ROOT = fallback
                break
_VENV_PYTHON = os.path.join(_KAI0_ROOT, '.venv', 'bin', 'python')
_VENV_PREFIX = os.path.join(_KAI0_ROOT, '.venv')

if (os.access(_VENV_PYTHON, os.X_OK)
        and os.path.abspath(sys.prefix) != os.path.abspath(_VENV_PREFIX)):
    # 当前不在 kai0 venv 中 (可能是 conda python3.13 或裸系统 python3.12)
    # 清理 PATH 中的 conda 路径, 防止 conda 的 libpython/importlib 污染 re-exec 后的进程
    _clean_path = ':'.join(p for p in os.environ.get('PATH', '').split(':')
                           if 'conda' not in p.lower())
    os.environ['PATH'] = _clean_path
    # 确保 LD_LIBRARY_PATH 也不含 conda
    _clean_ld = ':'.join(p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':')
                         if 'conda' not in p.lower())
    os.environ['LD_LIBRARY_PATH'] = _clean_ld
    # 用 venv python 重新启动自己, 保留所有命令行参数
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import time
import threading
import queue
from collections import deque

# 确保 openpi src 可被 import
_KAI0_SRC = os.path.join(_KAI0_ROOT, 'src')
if os.path.isdir(_KAI0_SRC) and _KAI0_SRC not in sys.path:
    sys.path.insert(0, _KAI0_SRC)

# 确保 CUDA 库路径在 JAX import 前设好
import glob as _glob
_venv_lib = os.path.join(_KAI0_ROOT, '.venv', 'lib')
_venv_pydirs = sorted(_glob.glob(os.path.join(_venv_lib, 'python3.*')))
_venv_sp = os.path.join(_venv_pydirs[-1], 'site-packages') if _venv_pydirs else os.path.join(_venv_lib, 'python3.12', 'site-packages')
_venv_nvidia = os.path.join(_venv_sp, 'nvidia')
_nvidia_libs = ':'.join(sorted(_glob.glob(os.path.join(_venv_nvidia, '*', 'lib'))))
if _nvidia_libs:
    os.environ['LD_LIBRARY_PATH'] = _nvidia_libs + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    import ctypes
    try:
        for lib_dir in _nvidia_libs.split(':'):
            for so in sorted(_glob.glob(os.path.join(lib_dir, '*.so*'))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    except Exception:
        pass

import select
import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Header
from piper_msgs.msg import PosCmd   # ③ 固件笛卡尔 IK: EE 位姿命令 → arm_reader EndPoseCtrl
from scipy.spatial.transform import Rotation as R_


class _PipeTrace:
    """Opt-in pipeline tracer (XVLA_TRACE_DIR) — client 侧, 与 serve_policy_xvla.py 同款。
    所有写盘 try/except 包裹: tracing 永不影响控制循环。trace 关时根本不构造。
    落盘 schema 见 xvla/analyze_pipeline_trace.py。"""

    def __init__(self, root, side):
        self.side = side
        self.arr_dir = os.path.join(root, f"{side}_arrays")
        self.img_dir = os.path.join(root, f"{side}_images")
        os.makedirs(self.arr_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        self._f = open(os.path.join(root, f"{side}_trace.jsonl"), "a", buffering=1)

    def event(self, **rec):
        rec.setdefault("t_wall", time.time())
        try:
            self._f.write(json.dumps(rec, default=float, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def save_arrays(self, seq, **arrs):
        try:
            np.savez_compressed(os.path.join(self.arr_dir, f"{int(seq):06d}.npz"), **arrs)
        except Exception:
            pass

    def save_image(self, seq, name, img):
        try:
            a = np.asarray(img)
            if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
                a = np.transpose(a, (1, 2, 0))
            cv2.imwrite(os.path.join(self.img_dir, f"{int(seq):06d}_{name}.jpg"),
                        cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2BGR))
        except Exception:
            pass


def _stamp_to_sec(stamp):
    """Convert a ROS2 stamp (sec + nanosec) to float seconds."""
    return stamp.sec + stamp.nanosec * 1e-9


def _to_bool(value) -> bool:
    """Convert a ROS2 parameter value to bool.

    LaunchConfiguration always passes strings, so 'false'/'true' must be
    handled explicitly — Python's bool('false') is True!
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', '1', 'yes')
    return bool(value)


# ────────────────────────────────────────────────────────────────────────────
# StreamActionBuffer — 从原版 agilex_inference_openpi_temporal_smoothing_ros2.py
#                      逐行复制, 不做任何修改
# ────────────────────────────────────────────────────────────────────────────
class StreamActionBuffer:
    """
    Maintains a queue of action chunks; each chunk is a deque([action0, action1, ...]).
    - New inferred chunks are appended from the right;
    - For each published step, popleft() the leftmost action from each chunk;
    - Empty chunks are dropped.
    """
    def __init__(self, max_chunks=10, decay_alpha=0.25, state_dim=14, smooth_method="temporal",
                 rot_euler_dims=None):
        self.chunks = deque()                 # Kept for backward compatibility
        self.max_chunks = max_chunks
        self.lock = threading.Lock()
        self.decay_alpha = float(decay_alpha)  # Smoothing strength (exponential weight)
        self.state_dim = state_dim
        self.smooth_method = smooth_method
        # chunk-overlap blend 时, 这些 [start,end) 维是 euler 角 (EE-firmware 的 pose14
        # rpy 段), 必须走测地 slerp 而非逐分量线性混合 — 否则相邻 chunk 的角度跨 ±180° wrap
        # 时线性平均会落到中间错误姿态 (实测左腕 roll 贴 ±180 → blend 出 ~0, 测地差 ~180°)。
        # joint 模式 = None → 关节角线性混合 (正确)。例: [(3,6),(10,13)]。
        self.rot_euler_dims = rot_euler_dims
        self.cur_chunk = deque()              # Current sequence to publish (after smoothing)
        self.k = 0                            # Published step count (for latency trimming)
        self.last_action = None               # Last successfully popped action

    def push_chunk(self, actions_chunk: np.ndarray):
        """Legacy interface (no longer used)."""
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            dq = deque([a.copy() for a in actions_chunk], maxlen=None)
            self.chunks.append(dq)
            while len(self.chunks) > self.max_chunks:
                self.chunks.popleft()

    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int, min_m: int = 8):
        """
        Integrate a new inference chunk:
        1) Trim the front of the new chunk by current k and max_k (latency compensation).
        2) If there is an existing chunk (cur_chunk), apply temporal smoothing on the overlap:
           - Overlap: first element 100% old / 0% new, last element 0% old / 100% new.
           - Extra tail from the new chunk is appended.
        3) Reset k=0 as the new current execution sequence.
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            max_k = max(0, int(max_k))
            min_m = max(1, int(min_m))
            drop_n = min(self.k, max_k)
            if drop_n >= len(actions_chunk):
                # Entire chunk trimmed; skip this update
                return
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]
            # Build old sequence: if empty but last_action exists, extend with last_action to min_m steps;
            # if non-empty and len < m, pad tail to min_m; if both empty, take new sequence as-is
            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    self.k = 0
                    return
            new_list = list(new_chunk)

            # Overlap length = min of remaining old length and new length
            overlap_len = min(len(old_list), len(new_list))
            if overlap_len <= 0:
                # No overlap; use new sequence as-is
                self.cur_chunk = deque(new_list, maxlen=None)
                self.k = 0
                return

            # If old sequence is longer than new, trim old tail
            if len(old_list) > len(new_list):
                old_list = old_list[:len(new_list)]
                overlap_len = len(new_list)

            # Weights for chunk-overlap blending. Default = linear (legacy).
            # Layer 1.1B = quintic smoothstep `6t^5 - 15t^4 + 10t^3` (minimum-jerk
            # transition: 1st + 2nd derivatives vanish at both endpoints — chunk
            # boundary continuity up to acceleration). LiPo arXiv:2506.05165.
            if overlap_len == 1:
                w_old = np.array([1.0], dtype=float)
            else:
                tau = np.linspace(0.0, 1.0, overlap_len, dtype=float)
                if self.smooth_method == 'min_jerk':
                    s = 6.0 * tau**5 - 15.0 * tau**4 + 10.0 * tau**3
                    w_old = 1.0 - s
                else:  # 'linear' (default, legacy)
                    w_old = 1.0 - tau
            w_new = 1.0 - w_old

            # C.3 2026-05-23: 矢量化 smooth (替 Python list comprehension).
            # 旧 for-loop 实测 25% cycle 出 P95 spike 2-8ms; vectorize 数学等价.
            # 真因后定为 Linux CFS 抢占 inference thread, vectorize 边际 -0.4ms;
            # 但保留 — 代码更简洁, 真修 P95 需 SCHED_FIFO (CAP_SYS_NICE).
            old_arr = np.asarray(old_list[:overlap_len], dtype=float)  # (overlap_len, 14)
            new_arr = np.asarray(new_list[:overlap_len], dtype=float)
            smoothed_arr = w_old[:, None] * old_arr + w_new[:, None] * new_arr
            # 姿态维改测地 slerp: R_blend = R_old · (R_old⁻¹·R_new)^w_new (沿最短弧插值,
            # 正确处理 ±180° wrap)。其余维 (xyz/gripper/关节) 保持上面的线性混合。
            if self.rot_euler_dims:
                for (a, b) in self.rot_euler_dims:
                    Ro = R_.from_euler('xyz', old_arr[:, a:b])
                    Rn = R_.from_euler('xyz', new_arr[:, a:b])
                    rel = (Ro.inv() * Rn).as_rotvec()                  # (overlap_len, 3) 最短弧
                    Rb = Ro * R_.from_rotvec(rel * w_new[:, None])     # 沿弧走 w_new 比例
                    smoothed_arr[:, a:b] = Rb.as_euler('xyz')
            smoothed = [smoothed_arr[i] for i in range(overlap_len)]
            # Append the extra tail from the new sequence
            combined = smoothed + new_list[overlap_len:]
            self.cur_chunk = deque([a.copy() for a in combined], maxlen=None)
            self.k = 0

    def has_any(self):
        with self.lock:
            return len(self.cur_chunk) > 0

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return the next action to publish; k += 1."""
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            # If about to pop the last element, save it as last_action
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self.k += 1
            return act

    def flush(self, seed_action: np.ndarray | None = None):
        """Discard all buffered actions and optionally seed for smooth blending."""
        with self.lock:
            self.cur_chunk.clear()
            self.k = 0
            self.last_action = seed_action


# ────────────────────────────────────────────────────────────────────────────
# ObsPrefetchWorker — A.2 流水线 (§7.9), 2026-05-23
# ────────────────────────────────────────────────────────────────────────────
# 在背景线程持续 pop _get_observation(), 把准备好的 obs 放进 maxsize=1 queue.
# 主推理线程从 queue.get() 拿 obs (几乎 0 等), 同时 worker 在 forward (~35ms) 期间
# 准备下一帧 obs (~15ms). 等于把 obs_construct 藏到 forward 背后, cycle 62 → ~44ms.
#
# Queue 策略: maxsize=1 + "drop old, put new" — 永远保持 queue 含最新可用 obs.
# 主线程取走后 worker 立刻开始下一轮; main 慢 worker 快 → worker 卡在 put 阻塞;
# main 快 worker 慢 → main fallback 走同步路径 (queue.Empty 异常).
class ObsPrefetchWorker:
    """Background thread that pre-fetches obs while main inference cycle runs.

    Usage:
        worker = ObsPrefetchWorker(owner=node)
        worker.start()  # daemon thread
        ...
        try:
            obs = worker.get_obs(timeout=0.1)  # main 推理 loop 取
        except queue.Empty:
            obs = node._get_observation()  # fallback sync
        ...
        worker.stop()  # on destroy
    """

    def __init__(self, owner, logger=None, queue_size: int = 1):
        self._owner = owner  # PolicyInferenceNode instance — calls _get_observation()
        self._logger = logger if logger is not None else owner.get_logger()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._running = False
        self._thread = None
        self._n_fetched = 0
        self._n_dropped = 0  # 主线程慢, queue full 时丢的旧 obs 数

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name='ObsPrefetchWorker')
        self._thread.start()
        self._logger.info('ObsPrefetchWorker started (A.2 §7.9 pipeline)')

    def stop(self):
        self._running = False
        # daemon thread, no join — just let it die with process

    def get_obs(self, timeout: float = 0.1):
        """Get freshest pre-fetched (obs, head_stamp_ns) tuple. Raises queue.Empty on timeout."""
        return self._queue.get(timeout=timeout)

    def _loop(self):
        # 等 policy 加载完毕 (sensor data ready). 跟 _inference_loop 同款.
        while self._running and self._owner.policy is None:
            time.sleep(0.05)

        while self._running:
            try:
                # C.2 2026-05-23: 用 _get_observation_with_stamp 拿 (obs, stamp_ns) tuple.
                # stamp 通过 queue 侧通道传给 main, 不进 obs dict — JAX/V1 server 看不到.
                obs, stamp_ns = self._owner._get_observation_with_stamp()
                if obs is None:
                    # sensor 数据不全 (sync_frame 失败), 短暂 yield 后重试
                    time.sleep(0.005)
                    continue

                item = (obs, stamp_ns)
                # Drop-old, put-new: queue 永远含最新 obs.
                # 若 main 还没取走上一个, 就 drop 上一个换新.
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    try:
                        _ = self._queue.get_nowait()
                        self._n_dropped += 1
                    except queue.Empty:
                        pass
                    try:
                        self._queue.put_nowait(item)
                    except queue.Full:
                        # race condition (main 同时也在取), 放弃这帧, 下轮重 fetch
                        pass
                self._n_fetched += 1

            except Exception as e:
                # 不能让 exception 杀掉 worker → main 永远拿不到 obs
                self._logger.warn(f'ObsPrefetchWorker fetch error: {e}')
                time.sleep(0.05)

        self._logger.info(
            f'ObsPrefetchWorker stopped (fetched={self._n_fetched}, '
            f'dropped_old={self._n_dropped})')


# ────────────────────────────────────────────────────────────────────────────
# PolicyInferenceNode
# ────────────────────────────────────────────────────────────────────────────
class PolicyInferenceNode(Node):
    """ROS2 node that integrates policy inference directly."""

    def __init__(self):
        super().__init__('policy_inference_node')

        # ── Parameters ──
        self.declare_parameter('mode', 'ros2')  # ros2 | websocket | both
        self.declare_parameter('config_name', 'pi05_flatten_fold_normal')
        self.declare_parameter('checkpoint_dir', '')
        self.declare_parameter('host', 'localhost')
        self.declare_parameter('port', 8000)
        self.declare_parameter('ws_port', 8000)
        # C.4 2026-05-23 (§7.8): transport 选择. default 'ws' = JAX legacy 行为
        # (TCP loopback + msgpack). 'shm' = POSIX shm (V1 path opt-in, -5-7ms cycle P95).
        self.declare_parameter('transport', 'ws')
        self.declare_parameter('shm_req_name', 'kai0_v1_obs')
        self.declare_parameter('shm_resp_name', 'kai0_v1_chunk')
        self.declare_parameter('prompt', 'Flatten and fold the cloth.')
        # Domain id for action-head-cond / soft-prompt models (Pi0Config.action_head_cond_num_domains>0).
        # -1 = disabled (obs carries no dataset_id → Observation.dataset_id=None, identical to plain pi05).
        # Set >=0 (e.g. 1=vis) to force the per-domain token at inference; plumbed from sidecar deploy_dataset_id.
        self.declare_parameter('dataset_id', -1)
        self.declare_parameter('publish_rate', 30)
        self.declare_parameter('inference_rate', 3.0)
        self.declare_parameter('chunk_size', 50)
        # ② 开环整 chunk (上游 SoftFold)。但主机 scipy IK ~0.4s + 30步快播下开环必见底卡顿,
        # 故默认【关】, 改走 RTC 式: 连续推理 (buffer 不见底) + overlap-blend + commanded-proprio。
        # 需要严格开环 (硬件够快/低 publish_rate) 时再开 open_loop_chunk:=true。
        self.declare_parameter('open_loop_chunk', False)
        self.declare_parameter('open_loop_min_remaining', 8)
        # RTC 式 commanded-proprio: execute 时用【上次下发的关节命令】当 proprio (而非实测),
        # 去传感噪声/臂滞后 → 连续推理下连续 chunk 一致 (上游平滑机制的"正确时序版": 喂刚下发
        # 的命令而非 chunk 末步, 连续模式不前瞻、不脱离进度)。images 仍为实测。
        self.declare_parameter('proprio_cmd_feedback', True)
        self.declare_parameter('latency_k', 8)
        self.declare_parameter('min_smooth_steps', 8)
        self.declare_parameter('decay_alpha', 0.25)
        # P2 Step 1+2 (2026-05-23, §7.8): client obs_construct fast path.
        # When True: skip JPEG mapping (#2, ~10-25ms) + CvBridge (#1, ~3-8ms) +
        #            BGR↔RGB cvtColor (~0.2ms). 用 np.frombuffer 直接 view msg.data;
        #            multi_camera_node 已 publish rgb8, 模型也要 rgb8, 全程零格式转换.
        # 默认 False = JAX legacy 行为 (bridge.imgmsg_to_cv2('bgr8') → _jpeg_mapping
        #                             → cv2.cvtColor(BGR2RGB) → resize_with_pad).
        # V1 路径通过 start_autonomy_v1.sh 传 'true' 启用.
        # 训练数据 loader 不做 JPEG (MP4 codec 是 AV1/h264, _jpeg_mapping 本来就没匹配
        # 训练 artifact), 跳过后模型输入实际更接近 "训练时 mp4-decode 输出".
        self.declare_parameter('fast_obs_pipeline', False)
        # A.2 异步流水线 (2026-05-23, §7.9): background ObsPrefetchWorker 跟主推理
        # forward 并行准备下一帧 obs, cycle 62→44ms (22.6Hz, 真机验证 ✓). JAX legacy
        # 默认 False. V1 路径 start_autonomy_v1.sh 传 'true' 启用.
        self.declare_parameter('pipelined_obs', False)
        # ── RTC (Real-Time Chunking) parameters ──
        # enable_rtc=True + Pi0Config → auto-upgraded to Pi0RTCConfig at load time
        # (same weights, model class swap only; see _load_jax_policy). Guidance is
        # applied inside sample_actions only when prev_action_chunk is not None,
        # so cold-start / flush / observe→execute transitions fall back to non-RTC.
        self.declare_parameter('enable_rtc', True)
        self.declare_parameter('rtc_execute_horizon', 16)
        self.declare_parameter('rtc_max_guidance_weight', 0.5)
        # obs 图像 resize 目标 (resize_with_pad). 默认 224x224 (openpi 模型输入, kai0 不变)。
        # gwp 世界模型 server 侧拼成 768x192, 需要近原生帧 → 用更大值 (如 480x640) 少损失。
        self.declare_parameter('obs_image_h', 224)
        self.declare_parameter('obs_image_w', 224)
        # Layer 1.1B — chunk overlap smoothing method. Default 'min_jerk' (quintic
        # smoothstep 6t^5 - 15t^4 + 10t^3, 1st+2nd derivatives vanish at endpoints →
        # minimum-jerk chunk-boundary transition; LiPo paper arXiv:2506.05165).
        # Validated 2026-05-25 on vis_v2_full real-machine: jiggle -46%, jerk peak/s
        # -41% vs 'linear', plus post-task attractor freeze (cmd converges to const
        # after task → 0 drift in idle scenes). Set to 'linear' to fall back to legacy.
        self.declare_parameter('rtc_smooth_method', 'min_jerk')
        # Layer 1.1E — publish-time EMA smoothing on cmd timeline (orthogonal to
        # 1.1B which smooths chunk boundaries). Default 1.0 = OFF (legacy/JAX path
        # bit-identical). The V1 path opts in via start_autonomy_v1.sh
        # (publish_smooth_alpha:=0.7). α∈(0,1]: cmd[t] := α*cmd + (1-α)*last_published.
        # Phase lag ≈ (1-α)/α timestep (α=0.7 @80Hz → 5.4ms; α=0.5 → 12.5ms; all
        # ≪ Piper PD ~100ms). Validated 2026-05-25 on vis_v2_full (α=0.5: state
        # jiggle -65%, jerk95_state -29%); α retuned to 0.7 pending re-validation.
        self.declare_parameter('publish_smooth_alpha', 1.0)
        # Experiment-1a (closed-loop oscillation fix): low-pass the STATE fed to
        # the model, to break the positive-feedback loop "state jitter → action
        # jitter → state jitter" that drives the ~1Hz limit-cycle observed on
        # A_0423_0527 (2026-05-29 diag). obs_state[t] := β·raw + (1-β)·prev_obs.
        # β=1.0 = OFF (obs bit-identical legacy). Gripper dims (6,13) are SKIPPED
        # (binary; smoothing traps half-grasp). Reset on observe→execute via flush.
        self.declare_parameter('obs_state_lowpass_alpha', 1.0)
        # NOTE: mask_prefix_delay is declared upstream in pi0_rtc.py:244 but
        # not exposed here — forwarding a Python bool through jit triggers
        # TracerBoolConversionError at pi0_rtc.py:323. Left as function default
        # (False) until the model uses jax.lax.cond for that branch.
        self.declare_parameter('gripper_offset', 0.003)
        # Gripper close-snap (autonomy-only, flag-gated, default ON). 位控夹爪在
        # action==state 数据上训练 → 抓软物只输出"布料厚度"对应的小开度 (≈5mm) →
        # 到位后位置误差≈0 → 保持力≈0 → 脱手. 把 RAW 模型夹爪输出 < thresh 的部分
        # 裁成 0 (软阈值: ≥ thresh 原样通过, 不动开侧/释放轨迹), 让固件保持位置误差
        # 顶到电流上限 → 夹紧力. 默认开 (5mm 阈值很保守, 只吃掉"已经基本闭合"的指令,
        # 对 action≠state 新模型几乎无副作用); 如需关闭设 gripper_close_snap:=false.
        # 作用于 joint 发布路径, v0/v1/v2 共用本节点 → 全链生效.
        self.declare_parameter('gripper_close_snap', True)
        self.declare_parameter('gripper_close_snap_thresh', 0.005)  # 5mm, 对齐模型 RAW 输出
        self.declare_parameter('img_front_topic', '/camera_f/camera/color/image_raw')
        self.declare_parameter('img_left_topic', '/camera_l/camera/color/image_raw')
        self.declare_parameter('img_right_topic', '/camera_r/camera/color/image_raw')
        self.declare_parameter('puppet_left_topic', '/puppet/joint_left')
        self.declare_parameter('puppet_right_topic', '/puppet/joint_right')
        self.declare_parameter('depth_front_topic', '/camera_f/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('gpu_id', 0)
        self.declare_parameter('execute_mode', False)
        self.declare_parameter('enable_rerun', False)
        self.declare_parameter('calibration_config', '')

        # ── EE-mode params (multimodal protocol §B.6 — for Cartesian-output models
        #    e.g. X-VLA; see docs/deployment/inference/xvla_inference_bringup.md §4 C)
        #    'joint'  : server emits [H,14] joint, publish directly (default, legacy).
        #    'ee_pose': server emits [H,16] world EE (action_kind="ee"); IK to joints
        #               (link6, R1) BEFORE buffering — downstream stays joint-domain. ──
        self.declare_parameter('execution_mode', 'joint')   # joint | ee_pose
        self.declare_parameter('urdf_path', '')             # auto-resolve calib/piper_local.urdf
        self.declare_parameter('calibration_yaml', '')      # auto-resolve config/calibration.yml
        # EE-mode IK 调参 (ee_pose only). 真机实测: 失败几乎全是位置够不到 (20-40mm),
        # 硬冻结→卡顿, 够不到时解翻分支→相邻步跳 ~100°. 见 xvla_inference_bringup §IK调参.
        #   策略: 纯位姿 IK(可达目标精确)+ 软接受(5mm<pr<=soft_pos 的最近可达解也用, 止抖)
        #         + 跳变拒绝(相对种子跳 >max_jump 视为翻分支→hold, 止姿态乱)。
        self.declare_parameter('ee_ik_seed_weight', 0.0)    # 种子软约束权重 (默认 0=纯位姿; >0 会损精度)
        self.declare_parameter('ee_ik_soft_pos', 0.04)      # 软接受位置阈值 m (超此判够不到→hold)
        self.declare_parameter('ee_ik_soft_rot', 0.12)      # 软接受姿态阈值 rad (~6.9°)
        self.declare_parameter('ee_ik_max_jump', 0.15)      # 关节单步限速 rad (~8.6°/step; clamp 防翻飞且不冻结)
        # ③ EE 控制后端: 'firmware'=发 PosCmd → 固件 EndPoseCtrl 做笛卡尔 IK (上游同款, 无主机 IK
        # → 无延迟/分支翻转/rate-limit 接缝踢, 丝滑); 'joint'=主机 PiperDHIK 反解到关节 (旧路径, 备用)。
        self.declare_parameter('ee_ctrl', 'firmware')
        # ③ EE 夹爪 re-binarize: server 端夹爪是二值的 {0=闭, open_m=张}, 但 stream_buffer 的
        # chunk 重叠平滑会把它线性混成中段值 (实测 0.076/0.077 等)。中段夹爪命令 = 物理上
        # "half-grasp 抓了又放" (见 _publish_action L2704 注释)。发 PosCmd 前 snap 回 {0, open_m}
        # 消除中段。阈值 = open_m/2。连续夹爪模型请置 False (默认 True: 当前 EE 仅 X-VLA, 二值)。
        self.declare_parameter('ee_gripper_binarize', True)
        self.declare_parameter('ee_gripper_open_m', 0.08)   # 张开物理行程 (m), 与 server gripper_open_value 对齐

        # ── Replay mode params (P1) ──
        # 'inference' = call policy.infer() (default, existing path)
        # 'replay'    = pop from preloaded parquet actions, no policy call
        # 'idle'      = block, no publishing (transition / fail-safe)
        self.declare_parameter('replay_mode', 'inference')
        self.declare_parameter('replay_episode_path', '')
        self.declare_parameter('replay_rate', 1.0)
        self.declare_parameter('replay_loop', False)
        # Auto-home: when current pose is > 5° off action[0], prepend a linear
        # interpolation segment so arm slow-walks from current → action[0] before
        # the recorded trajectory plays. Avoids requiring manual pre-alignment.
        self.declare_parameter('replay_auto_home', True)
        self.declare_parameter('replay_home_duration', 3.0)  # seconds, [1, 10]

        self.mode = self.get_parameter('mode').value
        self.prompt = self.get_parameter('prompt').value
        self._dataset_id = int(self.get_parameter('dataset_id').value)  # -1 = no per-domain token (plain pi05)
        self.publish_rate = self.get_parameter('publish_rate').value
        self.inference_rate = self.get_parameter('inference_rate').value
        self._open_loop_chunk = bool(self.get_parameter('open_loop_chunk').value)
        self._open_loop_min_remaining = int(self.get_parameter('open_loop_min_remaining').value)
        self._last_cycle_s = 0.6   # 实测整周期 (infer+IK+integrate) 时长, 自适应 gate 用; 初值保守
        self._proprio_cmd_fb = bool(self.get_parameter('proprio_cmd_feedback').value)
        self.chunk_size = self.get_parameter('chunk_size').value
        self.latency_k = self.get_parameter('latency_k').value
        self.min_smooth_steps = self.get_parameter('min_smooth_steps').value
        self.decay_alpha = self.get_parameter('decay_alpha').value
        self._enable_rtc = _to_bool(self.get_parameter('enable_rtc').value)
        self._rtc_execute_horizon = int(self.get_parameter('rtc_execute_horizon').value)
        self._rtc_max_guidance_weight = float(self.get_parameter('rtc_max_guidance_weight').value)
        self._obs_image_h = int(self.get_parameter('obs_image_h').value)
        self._obs_image_w = int(self.get_parameter('obs_image_w').value)
        self.gripper_offset = self.get_parameter('gripper_offset').value
        self._grip_close_snap = bool(self.get_parameter('gripper_close_snap').value)
        self._grip_close_snap_thr = float(self.get_parameter('gripper_close_snap_thresh').value)
        self._ee_grip_binarize = bool(self.get_parameter('ee_gripper_binarize').value)
        self._ee_grip_open_m = float(self.get_parameter('ee_gripper_open_m').value)
        # P2 Step 1+2: fast obs pipeline (skip JPEG + CvBridge + cvtColor)
        self._fast_obs_pipeline = _to_bool(self.get_parameter('fast_obs_pipeline').value)
        if self._fast_obs_pipeline:
            self.get_logger().info(
                'fast_obs_pipeline=True → bypass JPEG mapping + CvBridge + BGR↔RGB '
                '(rgb8 直传, np.frombuffer view msg.data; P2 §7.8)')
        # A.2 异步流水线 worker (§7.9): 仅在 pipelined_obs=True 时启动
        self._pipelined_obs = _to_bool(self.get_parameter('pipelined_obs').value)
        self._obs_prefetch = None  # 在 _infer_thread.start 之前 setup

        # ── Replay state ──
        self._replay_mode = str(self.get_parameter('replay_mode').value)
        self._replay_rate = max(0.5, min(1.5, float(self.get_parameter('replay_rate').value)))
        self._replay_loop = _to_bool(self.get_parameter('replay_loop').value)
        self._replay_auto_home = _to_bool(self.get_parameter('replay_auto_home').value)
        self._replay_home_duration = max(1.0, min(10.0, float(self.get_parameter('replay_home_duration').value)))
        self._replay_lock = threading.Lock()
        self._replay_actions = None    # np.ndarray[T, 14] when loaded, else None
        self._replay_path = ''
        self._replay_parquet_fps = float(self.publish_rate)  # measured from timestamps
        self._replay_buffer_total = 0  # total frames in cur_chunk after fps-comp + home prepend
        self._replay_aligned_threshold_rad = float(np.deg2rad(5.0))
        self._replay_progress_last_ts = 0.0  # rate-limit /replay_progress publish

        self.get_logger().info(f'Mode: {self.mode}')

        # ── State ──
        self.bridge = CvBridge()
        self.policy = None
        _smooth_method = str(self.get_parameter('rtc_smooth_method').value)
        # EE-firmware 模式 buffer 里跑的是 pose14 (含 euler rpy 两段) → 这两段必须 slerp,
        # 否则 chunk 边界跨 ±180° wrap 的线性混合会出错误姿态 (见 StreamActionBuffer.rot_euler_dims)。
        # joint 模式 (关节角) 维持 None = 全线性。
        _rot_dims = None
        if (str(self.get_parameter('execution_mode').value) == 'ee_pose'
                and str(self.get_parameter('ee_ctrl').value) == 'firmware'):
            _rot_dims = [(3, 6), (10, 13)]   # pose14: rpyL=[3:6], rpyR=[10:13]
        self.stream_buffer = StreamActionBuffer(
            decay_alpha=self.decay_alpha, state_dim=14, smooth_method=_smooth_method,
            rot_euler_dims=_rot_dims)
        self.get_logger().info(f'StreamActionBuffer smooth_method = {_smooth_method}'
                               f'{"  rot_slerp=ON (EE pose rpy)" if _rot_dims else ""}')
        # Layer 1.1E EMA on publish timeline (orthogonal to 1.1B chunk-edge smooth)
        self._publish_smooth_alpha = float(self.get_parameter('publish_smooth_alpha').value)
        if self._publish_smooth_alpha < 1.0:
            self.get_logger().info(f'EMA post-process active: alpha={self._publish_smooth_alpha}')
        # Exp-1a: obs-state low-pass (break closed-loop feedback). prev buffer reset on flush.
        self._obs_state_lp_alpha = float(self.get_parameter('obs_state_lowpass_alpha').value)
        self._obs_state_lp_prev = None
        if self._obs_state_lp_alpha < 1.0:
            self.get_logger().info(f'obs-state low-pass active: alpha={self._obs_state_lp_alpha} (break feedback loop)')

        # B1 client-side latency profile (opt-in via KAI0_LATENCY_PROFILE=1).
        # Writes one CSV row per inference cycle to /tmp/kai0_latency_<pid>.csv with
        # 11 columns: cycle_idx, t_image_age_ms (camera-to-ros lag), t_obs_construct_ms,
        # t_ws_full_rtt_ms (raw infer() call), t_ws_overhead_ms (= rtt - server_total),
        # server_preproc/state/infer/post/total_ms (from result['policy_timing']),
        # t_buffer_integrate_ms, t_loop_total_ms (incl all of above).
        # When V1 serve is the backend, this gives the full end-to-end picture
        # complementing server-side profile (B1 plan §7.3).
        self._lat_profile_fp = None
        self._lat_profile_cycle = 0
        if os.environ.get("KAI0_LATENCY_PROFILE"):
            try:
                lat_path = f"/tmp/kai0_latency_{os.getpid()}.csv"
                self._lat_profile_fp = open(lat_path, "w", buffering=1)  # line-buffered
                self._lat_profile_fp.write(
                    "cycle,t_image_age_ms,t_obs_construct_ms,t_ws_full_rtt_ms,t_ws_overhead_ms,"
                    "server_preproc_ms,server_state_encode_ms,server_infer_ms,server_postproc_ms,"
                    "server_total_ms,t_buffer_integrate_ms,t_loop_total_ms\n"
                )
                self.get_logger().info(f"B1 latency profile → {lat_path}")
            except Exception as e:
                self.get_logger().warn(f"B1 latency profile init failed: {e}")
                self._lat_profile_fp = None
        # ── RTC state ──
        # _rtc_prev_chunk: last chunk returned by inference, sent as prev_action_chunk
        #   to the next call for chunk-boundary guidance. Reset to None on flush /
        #   observe→execute / enable_rtc toggle to avoid dragging new predictions
        #   toward a stale trajectory.
        # _last_infer_ms: latest measured inference latency; converted to action-
        #   horizon steps for `inference_delay` (pi0_rtc.py:301).
        self._rtc_lock = threading.Lock()
        self._rtc_prev_chunk = None
        self._last_infer_ms = 0.0
        # _norm_action_{mean,std}: action norm_stats for RTC prev_chunk normalize
        # (used at inference_loop:2267). Init to None as default; _load_jax_policy
        # populates from <ckpt>/assets/<asset>/norm_stats.json if found, otherwise
        # remains None → pc_send = pc (raw). In websocket mode (mode=websocket,
        # connecting to external V1/JAX serve), defaults to None — server side
        # does the denormalize, so client RTC sends raw prev_chunk by default.
        self._norm_action_mean = None
        self._norm_action_std = None

        # ── Sensor deques (原版帧同步模式: 回调 append, get_synced_frame 消费) ──
        self._sensor_lock = threading.Lock()
        self._img_front_deque = deque(maxlen=200)
        self._img_left_deque = deque(maxlen=200)
        self._img_right_deque = deque(maxlen=200)
        self._joint_left_deque = deque(maxlen=200)
        self._joint_right_deque = deque(maxlen=200)

        # ── Execution control ──
        self._exec_lock = threading.Lock()
        self._execution_enabled = _to_bool(self.get_parameter('execute_mode').value)
        self._last_published_action = None  # for jump detection
        self._MAX_JOINT_JUMP_RAD = 0.5  # ~28.6°, reject actions with larger per-joint jump
        mode_str = 'EXECUTE' if self._execution_enabled else 'OBSERVE'
        self.get_logger().info(f'Execution mode: [{mode_str}]')

        # ── Subscribers ──
        # 执行控制 topic
        self.create_subscription(Bool, '/policy/execute', self._cb_execute, 1)
        # 图像: RELIABLE 匹配 RealSense v4.56+ 默认 QoS (RELIABLE + TRANSIENT_LOCAL)
        # BEST_EFFORT subscriber 无法接收 RELIABLE publisher 的消息
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        # P1.a 2026-05-23: 当 Rerun 关闭 (V1 路径默认 --no-rerun) 时, sensor subs 用同一个
        # ReentrantCallbackGroup → 配合 main() 的 MultiThreadedExecutor (留作未来), callback
        # 可在推理 cycle 中并发 fire, deque 保持 fresh, image_age 预期 -20-30ms (§7.7).
        # 当前 main() 仍用 SingleThreadedExecutor (MultiThreaded 实测 hang, 见 §7.8 表 B.2/B.3),
        # 所以这个 Reentrant 组在 SingleThreaded 下退化 = MutuallyExclusive = bit-identical baseline.
        # Rerun 开时 fallback 到默认 group — Rerun set_time+log 非原子, 多线程会破坏 timeline.
        _use_rerun_early = _to_bool(self.get_parameter('enable_rerun').value)
        self._sensor_cb_group = ReentrantCallbackGroup() if not _use_rerun_early else self.default_callback_group
        self.create_subscription(Image,
            self.get_parameter('img_front_topic').value,
            self._cb_img_front, img_qos, callback_group=self._sensor_cb_group)
        self.create_subscription(Image,
            self.get_parameter('img_left_topic').value,
            self._cb_img_left, img_qos, callback_group=self._sensor_cb_group)
        self.create_subscription(Image,
            self.get_parameter('img_right_topic').value,
            self._cb_img_right, img_qos, callback_group=self._sensor_cb_group)
        # Depth image for Rerun point cloud (head camera only, aligned to color)
        self.create_subscription(Image,
            self.get_parameter('depth_front_topic').value,
            self._cb_depth_front, img_qos, callback_group=self._sensor_cb_group)
        self.create_subscription(JointState,
            self.get_parameter('puppet_left_topic').value,
            self._cb_joint_left, 1000, callback_group=self._sensor_cb_group)
        self.create_subscription(JointState,
            self.get_parameter('puppet_right_topic').value,
            self._cb_joint_right, 1000, callback_group=self._sensor_cb_group)

        # ── Publishers ──
        self.pub_action = self.create_publisher(JointState, '/policy/actions', 10)
        self.pub_left = self.create_publisher(JointState, '/master/joint_left', 10)
        self.pub_right = self.create_publisher(JointState, '/master/joint_right', 10)
        # Full predicted action chunk for visualization (50 steps × 14 dims)
        from std_msgs.msg import Float32MultiArray
        self.pub_action_chunk = self.create_publisher(
            Float32MultiArray, '/policy/action_chunk', 5)
        # Replay progress: [frame_idx, total_frames, done_flag]
        self.pub_replay_progress = self.create_publisher(
            Float32MultiArray, '/replay_progress', 5)

        # ── EE-mode kinematics (only when execution_mode=ee_pose) ──
        # Lives here so the legacy joint path is byte-identical when off (no IK,
        # no extra publishers, no PiperIK import). See _setup_ee_kinematics.
        self._execution_mode = self.get_parameter('execution_mode').value
        self._ik = None            # calib.piper_dh_ik.PiperDHIK (CalFK DH link6, 训练同模型)
        self._T_world_baseL = None
        self._T_world_baseR = None
        self._ee_T_world_baseL = None  # ee-path dedicated (decoupled from rerun-viz attrs above)
        self._ee_T_world_baseR = None
        self._ee_last_good = None  # [14] last successful IK solve (hold-on-failure seed)
        self._ee_ctrl = 'joint'    # ③ 默认值; ee_pose 时在 _setup_ee_kinematics 读真值
        self.pub_ee_left = None
        self.pub_ee_right = None
        self.pub_pos_left = None   # ③ PosCmd → /pos_cmd_{left,right} → 固件 EndPoseCtrl
        self.pub_pos_right = None
        # pipeline trace (opt-in): XVLA_TRACE_DIR 置位才落盘 (env 由 start_xvla_stack.sh 设,
        # 经 ros2 launch 继承到本节点)。关时 self._trace=None → 所有 trace 调用直接跳过。
        self._trace = None
        self._trace_seq = 0
        _tdir = os.environ.get('XVLA_TRACE_DIR')
        if _tdir:
            try:
                self._trace = _PipeTrace(_tdir, 'client')
                self.get_logger().warn(f'[trace] pipeline trace ON → {_tdir}')
            except Exception as e:
                self.get_logger().warn(f'[trace] init failed: {e}')
        if self._execution_mode == 'ee_pose':
            self._setup_ee_kinematics()

        # ── Rerun visualization (conditional) ──
        # Rerun set_time+log is non-atomic. All ROS2 callbacks/timers MUST share
        # a single thread. _rr_require_single_thread is checked in on_configure
        # and main() to enforce this at runtime, not just via comments.
        self._rr_require_single_thread = False
        self._use_rerun = _to_bool(self.get_parameter('enable_rerun').value)
        self._rr = None  # rerun module, lazy import
        self._rr_rec = None  # rerun RecordingStream
        self._rr_bridge = None  # separate CvBridge for Rerun callbacks (thread safety)
        self._fk = None
        self._T_world_baseL = None
        self._T_world_baseR = None
        self._T_world_camF = None
        self._cam_f_fx = self._cam_f_fy = self._cam_f_cx = self._cam_f_cy = None
        self._actual_trail_left = deque(maxlen=300)
        self._actual_trail_right = deque(maxlen=300)
        self._rr_frame_idx_left = 0
        self._rr_frame_idx_right = 0
        self._rr_fk_skip = 3  # only run FK visualization every N-th joint callback
        self._rr_fk_queue = deque(maxlen=60)  # (arm_label, q, T_base, stamp) for off-callback FK
        self._rr_fk_drops = 0  # count of dropped FK frames due to queue overflow
        self._rr_pred_queue = deque(maxlen=2)  # (actions, stamp) for off-thread predicted traj FK
        self._rr_pred_drops = 0  # count of dropped predicted trajectory frames
        # Queue for Rerun events from non-executor threads (keyboard, etc.)
        # Entries: (entity: str, data_factory: callable, stamp_sec: float)
        self._rr_event_queue = deque(maxlen=20)
        if self._use_rerun:
            self._init_rerun()
            # Rerun timer — processes FK queue, predicted traj, and cross-thread events
            if self._use_rerun:
                self._rr_require_single_thread = True
                self.create_timer(1.0 / 10.0, self._rr_process_fk_queue)  # ~10Hz

        # ── Load policy ──
        self._load_policy()

        # ── Keyboard listener thread ──
        self._kb_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._kb_thread.start()

        # ── A.2 异步 obs 流水线 worker (§7.9, 启动顺序: 先 worker, 再 inference 线程) ──
        if self._pipelined_obs:
            self._obs_prefetch = ObsPrefetchWorker(owner=self)
            self._obs_prefetch.start()
            self.get_logger().info(
                'pipelined_obs=True → obs_construct 与 forward 并行 (cycle 预期 -14ms)')

        # ── Inference thread ──
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_thread.start()

        # ── Publish timer ──
        period = 1.0 / self.publish_rate
        self.create_timer(period, self._publish_action)

        # ── Hot-reload callback for ros2 param set ──
        # Without this, rtc_apply.sh can change the declared parameter values
        # but the inference loop keeps reading the instance variables cached
        # from __init__, so runtime changes silently no-op.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info('Policy inference node ready')

    # ── Parameter hot-reload ────────────────────────────────────────

    def _on_set_parameters(self, params):
        """Apply ros2 param set changes to instance variables in-place.

        Called synchronously on the executor thread when a parameter is updated
        (e.g. via `ros2 param set /policy_inference enable_rtc false`). We only
        mirror the subset that the inference/publish loops actually read.
        """
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            try:
                if p.name == 'enable_rtc':
                    new_val = _to_bool(p.value)
                    if new_val != self._enable_rtc:
                        # Clear prev_chunk on toggle so guidance doesn't use stale data
                        with self._rtc_lock:
                            self._rtc_prev_chunk = None
                    self._enable_rtc = new_val
                    self.get_logger().info(f'enable_rtc → {new_val}')
                elif p.name == 'rtc_execute_horizon':
                    self._rtc_execute_horizon = int(p.value)
                    self.get_logger().info(f'rtc_execute_horizon → {p.value}')
                elif p.name == 'rtc_max_guidance_weight':
                    self._rtc_max_guidance_weight = float(p.value)
                elif p.name == 'inference_rate':
                    self.inference_rate = float(p.value)
                    self.get_logger().info(f'inference_rate → {p.value}')
                elif p.name == 'latency_k':
                    self.latency_k = int(p.value)
                elif p.name == 'min_smooth_steps':
                    self.min_smooth_steps = int(p.value)
                elif p.name == 'obs_state_lowpass_alpha':
                    self._obs_state_lp_alpha = float(p.value)
                    self._obs_state_lp_prev = None
                    self.get_logger().info(f'obs_state_lowpass_alpha → {self._obs_state_lp_alpha}')
                elif p.name == 'decay_alpha':
                    self.decay_alpha = float(p.value)
                    self.stream_buffer.decay_alpha = float(p.value)
                elif p.name == 'gripper_offset':
                    self.gripper_offset = float(p.value)
                elif p.name == 'gripper_close_snap':
                    self._grip_close_snap = bool(p.value)
                    self.get_logger().info(f'gripper_close_snap → {self._grip_close_snap}')
                elif p.name == 'gripper_close_snap_thresh':
                    self._grip_close_snap_thr = float(p.value)
                    self.get_logger().info(f'gripper_close_snap_thresh → {self._grip_close_snap_thr}')
                elif p.name == 'prompt':
                    self.prompt = str(p.value)
                elif p.name == 'replay_episode_path':
                    new_path = str(p.value)
                    if not new_path:
                        with self._replay_lock:
                            self._replay_actions = None
                            self._replay_path = ''
                        self.get_logger().info('[REPLAY] cleared episode path')
                    else:
                        ok, msg = self._load_replay_episode(new_path)
                        if not ok:
                            return SetParametersResult(successful=False, reason=msg)
                elif p.name == 'replay_rate':
                    rate = max(0.5, min(1.5, float(p.value)))
                    self._replay_rate = rate
                    self.get_logger().info(f'replay_rate → {rate}')
                elif p.name == 'replay_loop':
                    self._replay_loop = _to_bool(p.value)
                    self.get_logger().info(f'replay_loop → {self._replay_loop}')
                elif p.name == 'replay_auto_home':
                    self._replay_auto_home = _to_bool(p.value)
                    self.get_logger().info(f'replay_auto_home → {self._replay_auto_home}')
                elif p.name == 'replay_home_duration':
                    self._replay_home_duration = max(1.0, min(10.0, float(p.value)))
                    self.get_logger().info(f'replay_home_duration → {self._replay_home_duration}s')
                elif p.name == 'replay_mode':
                    new_rm = str(p.value)
                    if new_rm not in ('inference', 'replay', 'idle'):
                        return SetParametersResult(
                            successful=False,
                            reason=f'invalid replay_mode {new_rm!r}, expect inference|replay|idle')
                    if new_rm == 'replay':
                        # ALWAYS re-enter (re-fill buffer + re-check pose) even if
                        # mode was already 'replay'. Setting replay_mode=replay is
                        # the user's "restart this episode" trigger; short-circuit
                        # would leave a previous run's drained buffer in place →
                        # next /policy/execute=true sees remaining=0 → instant done.
                        ok, reason = self._enter_replay_mode()
                        if not ok:
                            return SetParametersResult(
                                successful=False,
                                reason=f'replay pre-flight failed: {reason}')
                    elif new_rm != self._replay_mode and self._replay_mode == 'replay':
                        # leaving replay (replay → inference|idle)
                        self._exit_replay_mode()
                    self._replay_mode = new_rm
                    self.get_logger().info(f'replay_mode → {new_rm}')
            except Exception as e:
                return SetParametersResult(successful=False, reason=f'{p.name}: {e}')
        return SetParametersResult(successful=True)

    # ── Replay mode (P1) ────────────────────────────────────────────

    def _load_replay_episode(self, parquet_path: str):
        """Load `action` column from a LeRobot v2.1 episode parquet, plus the real
        recording FPS inferred from timestamp deltas. Returns (ok: bool, msg: str).
        Stores result in self._replay_actions and self._replay_parquet_fps."""
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            return False, f'pyarrow not installed: {e}'
        if not os.path.isfile(parquet_path):
            return False, f'parquet not found: {parquet_path}'
        try:
            tbl = pq.read_table(parquet_path, columns=['action', 'timestamp'])
        except Exception as e:
            return False, f'parquet read failed: {e}'
        try:
            actions_list = tbl.column('action').to_pylist()
            actions = np.asarray(actions_list, dtype=np.float32)
        except Exception as e:
            return False, f'action column decode failed: {e}'
        if actions.ndim != 2 or actions.shape[1] != 14:
            return False, f'bad action shape {actions.shape}, expected [T, 14]'
        # Compute real recording FPS from timestamps. meta.json's `fps` field is
        # often inaccurate (writer-side declared, not measured), so we use the
        # actual deltas. This matters: replay was naively assuming 30Hz but
        # Task_P/.../episode_42 is actually 14Hz → naive replay played 2.14× speed.
        try:
            ts = np.asarray(tbl.column('timestamp').to_pylist(), dtype=np.float64)
            if len(ts) >= 2:
                duration_s = float(ts[-1] - ts[0])
                parquet_fps = (len(ts) - 1) / duration_s if duration_s > 1e-6 else float(self.publish_rate)
            else:
                duration_s = 0.0
                parquet_fps = float(self.publish_rate)
        except Exception:
            duration_s = actions.shape[0] / float(self.publish_rate)
            parquet_fps = float(self.publish_rate)
        with self._replay_lock:
            self._replay_actions = actions
            self._replay_path = parquet_path
            self._replay_parquet_fps = parquet_fps
        self.get_logger().info(
            f'[REPLAY] loaded {parquet_path}: {actions.shape[0]} frames, '
            f'duration={duration_s:.2f}s, parquet_fps={parquet_fps:.2f} Hz '
            f'(publish_rate={self.publish_rate} Hz)')
        return True, f'loaded {actions.shape[0]} frames @ {parquet_fps:.1f} Hz'

    def _check_start_pose_aligned(self):
        """Compare latest joint state to action[0] of currently loaded episode.
        Returns (aligned: bool, max_diff_rad: float, per_joint_diff_rad: list[float]).
        FAIL CLOSED on missing data (returns aligned=False)."""
        with self._replay_lock:
            if self._replay_actions is None:
                return False, float('inf'), []
            target = self._replay_actions[0].copy()
        with self._sensor_lock:
            jl = self._joint_left_deque[-1] if self._joint_left_deque else None
            jr = self._joint_right_deque[-1] if self._joint_right_deque else None
        if jl is None or jr is None:
            return False, float('inf'), []
        ql = np.asarray(jl.position[:7], dtype=np.float32)
        qr = np.asarray(jr.position[:7], dtype=np.float32)
        if len(ql) < 7 or len(qr) < 7:
            return False, float('inf'), []
        current = np.concatenate([ql, qr])
        diff = np.abs(current - target)
        max_diff = float(diff.max())
        aligned = bool(max_diff <= self._replay_aligned_threshold_rad)
        return aligned, max_diff, [float(x) for x in diff]

    def _verify_no_publisher_conflict(self):
        """Check no other node is publishing to /master/joint_left.
        Returns (ok: bool, conflict_node_names: list[str]). FAIL CLOSED.

        `ros2 topic info -v` output format (jazzy): no `Publishers:` / `Subscriptions:`
        section headers — instead each endpoint is its own block with
        `Endpoint type: PUBLISHER|SUBSCRIPTION`. Parse by splitting on blocks and
        filtering by Endpoint type. Earlier impl used a 'Publishers:' header regex
        that never matched, so ALL Node names (including subscribers) got flagged
        as publishers — false positives blocked legitimate replays."""
        try:
            import subprocess
            import re
            out = subprocess.run(
                ['ros2', 'topic', 'info', '/master/joint_left', '-v'],
                capture_output=True, text=True, timeout=5)
            if out.returncode != 0:
                return False, [f'ros2 topic info rc={out.returncode}: {out.stderr.strip()[:200]}']
            text = out.stdout
            # Split into per-endpoint blocks. Each block starts with "Node name:".
            blocks = re.split(r'\n(?=Node name:)', text)
            publishers = []
            for blk in blocks:
                if 'Endpoint type: PUBLISHER' not in blk:
                    continue
                m = re.search(r'Node name:\s*(\S+)', blk)
                if m:
                    publishers.append(m.group(1))
            my_name = self.get_name()
            others = [p for p in publishers if p != my_name]
            self.get_logger().info(
                f'[REPLAY] /master/joint_left publishers found={publishers}, '
                f'my_name={my_name!r}, conflicting={others}')
            return len(others) == 0, others
        except Exception as e:
            return False, [f'check_failed: {e}']

    def _verify_deployment_marker(self):
        """Check /tmp/kai0_deployment_mode == 'autonomy'.
        Returns (ok: bool, current_value_or_reason: str). FAIL CLOSED."""
        marker = '/tmp/kai0_deployment_mode'
        if not os.path.isfile(marker):
            return False, f'{marker} missing — start_autonomy.sh not active?'
        try:
            with open(marker) as f:
                val = f.read().strip()
        except Exception as e:
            return False, f'{marker} read failed: {e}'
        if val != 'autonomy':
            return False, f'{marker}={val!r}, replay requires autonomy'
        return True, val

    def _resample_actions(self, actions: np.ndarray, rate: float) -> np.ndarray:
        """Linear-interp resample [T, 14] by `rate` (no clamp here; user-facing
        replay_rate clamp happens at param boundary). rate < 1.0 → upsample
        (more frames, slower playback); rate > 1.0 → downsample (faster)."""
        rate = float(rate)
        if abs(rate - 1.0) < 1e-3:
            return actions
        T = actions.shape[0]
        new_T = max(1, int(round(T / rate)))
        if new_T == T:
            return actions
        old_idx = np.linspace(0.0, T - 1.0, T)
        new_idx = np.linspace(0.0, T - 1.0, new_T)
        return np.array(
            [np.interp(new_idx, old_idx, actions[:, d]) for d in range(actions.shape[1])],
            dtype=np.float32).T

    def _enter_replay_mode(self):
        """Run all pre-flight gates, fill stream_buffer, ready for execution.
        Caller must have already loaded _replay_actions via replay_episode_path param.
        Returns (ok: bool, reason_if_fail: str)."""
        with self._replay_lock:
            if self._replay_actions is None:
                return False, 'no episode loaded (set replay_episode_path first)'
        # S1: deployment marker
        ok, reason = self._verify_deployment_marker()
        if not ok:
            return False, f'deployment marker: {reason}'
        # S2: publisher conflict
        ok, others = self._verify_no_publisher_conflict()
        if not ok:
            return False, f'topic /master/joint_left has other publishers: {others}'
        # S4 + fps compensation pipeline. Two segments need DIFFERENT resampling:
        #   - Episode actions: recorded at parquet_fps (often != publish_rate); must
        #     be upsampled to publish_rate so wall-clock duration matches recording.
        #   - Auto-home interp: constructed at publish_rate spacing (we choose
        #     home_n based on replay_home_duration*publish_rate); already correct,
        #     no further resampling.
        # Mistake to avoid: resampling the concatenated [home + episode] uniformly
        # stretches the home segment too, breaking timing.
        publish_rate = float(self.publish_rate)
        parquet_fps = float(self._replay_parquet_fps)
        episode_eff_rate = self._replay_rate * (parquet_fps / publish_rate)
        if episode_eff_rate < 0.05 or episode_eff_rate > 3.0:
            self.get_logger().warn(
                f'[REPLAY] episode effective resample rate {episode_eff_rate:.3f} clamped — '
                f'replay_rate={self._replay_rate}, parquet_fps={parquet_fps}, '
                f'publish_rate={publish_rate}')
            episode_eff_rate = max(0.05, min(3.0, episode_eff_rate))
        if abs(episode_eff_rate - 1.0) > 1e-3:
            episode_actions = self._resample_actions(self._replay_actions, episode_eff_rate)
        else:
            episode_actions = self._replay_actions

        # S4: start pose alignment (with optional auto-home interpolation)
        aligned, max_d, per_joint = self._check_start_pose_aligned()
        home_n = 0
        home_frames = None
        if not aligned:
            if not self._replay_auto_home:
                deg_per = ', '.join(f'{np.rad2deg(x):.1f}' for x in per_joint)
                return False, (f'start pose not aligned: max_Δ={np.rad2deg(max_d):.2f}°, '
                               f'threshold=5°. per-joint(°)=[{deg_per}]')
            # AUTO-HOME at publish_rate (no further resample): home_duration_s × publish_rate frames.
            with self._sensor_lock:
                jl = self._joint_left_deque[-1] if self._joint_left_deque else None
                jr = self._joint_right_deque[-1] if self._joint_right_deque else None
            if jl is None or jr is None:
                return False, 'auto_home: missing joint_state for current pose'
            ql = np.asarray(jl.position[:7], dtype=np.float32)
            qr = np.asarray(jr.position[:7], dtype=np.float32)
            if len(ql) < 7 or len(qr) < 7:
                return False, 'auto_home: incomplete joint_state'
            current = np.concatenate([ql, qr])
            target = episode_actions[0]   # post-resample first frame == _replay_actions[0]
            home_n = max(1, int(round(self._replay_home_duration * publish_rate)))
            per_step_max = float(np.max(np.abs(target - current))) / home_n
            if per_step_max > self._MAX_JOINT_JUMP_RAD:
                return False, (f'auto_home: per-step Δ={np.rad2deg(per_step_max):.2f}° > '
                               f'{np.rad2deg(self._MAX_JOINT_JUMP_RAD):.0f}° jump-protect '
                               f'threshold. Increase replay_home_duration (currently '
                               f'{self._replay_home_duration:.1f}s).')
            alphas = np.linspace(1.0/home_n, 1.0, home_n, dtype=np.float32).reshape(-1, 1)
            home_frames = ((1.0 - alphas) * current[None, :] +
                           alphas * target[None, :]).astype(np.float32)
            self.get_logger().info(
                f'[REPLAY] auto-home: {home_n} interp frames '
                f'({self._replay_home_duration:.1f}s @{publish_rate:.0f}Hz). '
                f'initial max_Δ={np.rad2deg(max_d):.2f}°, '
                f'per-step Δ={np.rad2deg(per_step_max):.3f}°')

        # Concatenate (home is at publish_rate spacing already; episode just upsampled to it)
        if home_frames is not None:
            actions = np.vstack([home_frames, episode_actions]).astype(np.float32)
        else:
            actions = np.asarray(episode_actions, dtype=np.float32)

        with self.stream_buffer.lock:
            self.stream_buffer.cur_chunk = deque([a.copy() for a in actions], maxlen=None)
            self.stream_buffer.k = 0
            self.stream_buffer.last_action = None
        with self._replay_lock:
            self._replay_buffer_total = int(actions.shape[0])
        episode_after = actions.shape[0] - home_n
        wall_play_s = actions.shape[0] / publish_rate
        self.get_logger().info(
            f'[REPLAY] entered replay mode: buffer={actions.shape[0]} frames '
            f'(home={home_n} + episode={episode_after}), '
            f'replay_rate={self._replay_rate}, episode_resample={episode_eff_rate:.3f}, '
            f'parquet_fps={parquet_fps:.2f}→publish_rate={publish_rate:.0f}, '
            f'wall_clock_play={wall_play_s:.2f}s '
            f'(home={home_n/publish_rate:.2f}s + episode={episode_after/publish_rate:.2f}s), '
            f'loop={self._replay_loop}, '
            f'start_align_max_Δ={np.rad2deg(max_d):.2f}°{" (aligned)" if home_n == 0 else " (auto-home)"}')
        return True, ''

    def _exit_replay_mode(self):
        """Idempotent. Flush buffer + auto-disable execution (safety)."""
        self.stream_buffer.flush(seed_action=self._last_published_action)
        with self._exec_lock:
            self._execution_enabled = False
        self.get_logger().info('[REPLAY] exited replay mode (buffer flushed, execution OFF)')

    def _tick_replay(self):
        """Called per inference-loop iteration while replay_mode='replay'.
        Tracks progress, publishes /replay_progress, handles end-of-episode."""
        with self._replay_lock:
            actions = self._replay_actions
            total = self._replay_buffer_total  # post-resample + home, set by _enter_replay_mode
        if actions is None or total <= 0:
            return
        with self.stream_buffer.lock:
            remaining = len(self.stream_buffer.cur_chunk)
        idx = max(0, total - remaining)
        done = (remaining == 0)
        # Rate-limit progress publish (~10 Hz)
        from std_msgs.msg import Float32MultiArray
        now = time.monotonic()
        if now - self._replay_progress_last_ts >= 0.1 or done:
            msg = Float32MultiArray()
            msg.data = [float(idx), float(total), 1.0 if done else 0.0]
            try:
                self.pub_replay_progress.publish(msg)
            except Exception as e:
                self.get_logger().debug(f'replay_progress publish failed: {e}')
            self._replay_progress_last_ts = now
            # Per-second info log: includes execution_enabled to detect cases where
            # the buffer is full but execute=false (so nothing actually publishes).
            if not hasattr(self, '_replay_info_last_ts'):
                self._replay_info_last_ts = 0.0
            if now - self._replay_info_last_ts >= 1.0 or done:
                with self._exec_lock:
                    enabled = self._execution_enabled
                self.get_logger().info(
                    f'[REPLAY] frame {idx}/{total} ({100*idx/max(1,total):.1f}%) '
                    f'remaining={remaining} execute={enabled}')
                self._replay_info_last_ts = now
        # End-of-episode handling
        if done:
            if self._replay_loop:
                ok, reason = self._enter_replay_mode()
                if not ok:
                    self.get_logger().warn(f'[REPLAY] loop end: {reason}, stopping')
                    with self._exec_lock:
                        self._execution_enabled = False
                    self._replay_mode = 'inference'  # auto-revert on loop fail
            else:
                self.get_logger().info('[REPLAY] episode finished, execution OFF, mode→inference')
                with self._exec_lock:
                    self._execution_enabled = False
                # Auto-revert to inference mode at end-of-episode. Without this,
                # the inference loop keeps spinning in _tick_replay re-publishing
                # the same done=true /replay_progress msg forever, AND a follow-up
                # `replay_mode=replay` set would short-circuit (kept here as defense).
                self._replay_mode = 'inference'

    # ── Policy loading ──────────────────────────────────────────────

    def _load_policy(self):
        """Load policy based on mode."""
        if self.mode in ('ros2', 'both'):
            self._load_jax_policy()
            if self.mode == 'both':
                self._start_ws_server()
        elif self.mode == 'websocket':
            self._load_ws_policy()

    def _load_jax_policy(self):
        """Load JAX model directly (no WebSocket)."""
        config_name = self.get_parameter('config_name').value
        checkpoint_dir = self.get_parameter('checkpoint_dir').value
        gpu_id = str(self.get_parameter('gpu_id').value)

        if not checkpoint_dir:
            raise ValueError(
                'checkpoint_dir is required for ros2/both mode. '
                'Set via: -p checkpoint_dir:=gs://openpi-assets/checkpoints/pi05_base/params')

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', '/tmp/xla_cache')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['JAX_COMPILATION_CACHE_DIR'] = cache_dir

        # GPU memory: grow-on-demand instead of JAX's default upfront grab.
        # JAX defaults to PREALLOCATE=true + MEM_FRACTION=0.75, i.e. it reserves
        # 0.75×32GB ≈ 24GB the instant the GPU is first touched. On sim01 every
        # card is shared, so free VRAM is routinely < 24GB → that reservation
        # blocks indefinitely and create_trained_policy hangs forever right
        # after the Pi0→Pi0RTCConfig upgrade (observed 2026-06-05: dagger web
        # "Start session" never completes; reproduced on GPU 0 AND a 23.8GB-free
        # GPU 3). Set here (before any JAX import) rather than only via the
        # launch file, because ros2 launch SetEnvironmentVariable did not reach
        # this subprocess. setdefault → an explicit launch/shell override wins.
        # pi05 inference needs ~10-13GB, so 0.35 (≈11GB) is ample.
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.35')

        # 确保 CUDA 库路径 (reuse auto-detected site-packages path)
        venv_nvidia = os.path.join(_venv_sp, 'nvidia')
        if os.path.exists(venv_nvidia):
            cuda_libs = ':'.join(
                os.path.join(venv_nvidia, d, 'lib')
                for d in os.listdir(venv_nvidia)
                if os.path.isdir(os.path.join(venv_nvidia, d, 'lib'))
            )
            os.environ['LD_LIBRARY_PATH'] = cuda_libs + ':' + os.environ.get('LD_LIBRARY_PATH', '')

        self.get_logger().info(f'Loading JAX policy: config={config_name}, ckpt={checkpoint_dir}, GPU={gpu_id}')
        t0 = time.monotonic()

        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        from openpi.models import pi0_config as _pi0_config
        import dataclasses as _dc

        train_config = _config.get_config(config_name)

        # Auto-upgrade Pi0Config → Pi0RTCConfig when enable_rtc is on. Pi0RTCConfig
        # inherits from Pi0Config with no added fields, so this is a pure class
        # swap — same weights, same architecture, same transforms. Guidance logic
        # in sample_actions only engages when prev_action_chunk is not None, so
        # flipping this without prev_chunk is a no-op at inference time.
        if self._enable_rtc:
            base = train_config.model
            if isinstance(base, _pi0_config.Pi0RTCConfig):
                self.get_logger().info('Model already Pi0RTCConfig, no swap needed')
            elif isinstance(base, _pi0_config.Pi0Config):
                rtc_model = _pi0_config.Pi0RTCConfig(**{
                    f.name: getattr(base, f.name) for f in _dc.fields(base)
                })
                train_config = _dc.replace(train_config, model=rtc_model)
                self.get_logger().info(
                    f'Upgraded model: Pi0Config → Pi0RTCConfig (pi05={base.pi05}) '
                    f'for RTC guidance')
            else:
                self.get_logger().warn(
                    f'enable_rtc=True but model is {type(base).__name__} '
                    f'(not Pi0Config); skipping RTC swap, guidance will be inactive')
                self._enable_rtc = False

        self.policy = _policy_config.create_trained_policy(
            train_config, checkpoint_dir)

        self.get_logger().info(f'JAX policy loaded in {time.monotonic()-t0:.1f}s')

        # ── Load action norm_stats for RTC prev_chunk normalization (R1 fix) ──
        # Policy.infer extracts obs['prev_action_chunk'] and forwards it to
        # sample_actions WITHOUT running input_transform on it (policy.py:85-86).
        # But the diffusion trajectory x_t inside sample_actions lives in
        # NORMALIZED model space (mean=0, std=1 after Normalize transform).
        # If we send raw joint angles, the guidance error (prev - x_1) is
        # systematically biased by the un-normalization mean shift — upstream
        # agilex/arx RTC clients have the same bug. We normalize the 14-dim
        # prev_chunk here using actions.mean/std from norm_stats.json so
        # guidance compares like-for-like.
        import glob as _glob2
        self._norm_action_mean = None
        self._norm_action_std = None
        try:
            ns_candidates = _glob2.glob(os.path.join(checkpoint_dir, 'assets', '**', 'norm_stats.json'),
                                        recursive=True)
            if ns_candidates:
                import json as _json
                with open(ns_candidates[0]) as _f:
                    ns = _json.load(_f).get('norm_stats', {})
                acts = ns.get('actions', {})
                mean_full = np.asarray(acts.get('mean', []), dtype=np.float32)
                std_full = np.asarray(acts.get('std', []), dtype=np.float32)
                if mean_full.size >= 14 and std_full.size >= 14:
                    self._norm_action_mean = mean_full[:14].copy()
                    # Guard std against division by zero (pad dims would be 0)
                    self._norm_action_std = np.where(std_full[:14] < 1e-6, 1.0, std_full[:14]).astype(np.float32)
                    self.get_logger().info(
                        f'RTC normalize: loaded norm_stats from {ns_candidates[0]} | '
                        f'mean[:4]={self._norm_action_mean[:4].round(3).tolist()} '
                        f'std[:4]={self._norm_action_std[:4].round(3).tolist()}')
                else:
                    self.get_logger().warn(
                        f'norm_stats actions.mean has only {mean_full.size} dims (<14); '
                        f'RTC prev_chunk will be sent raw (no normalize)')
            else:
                self.get_logger().warn(
                    f'norm_stats.json not found under {checkpoint_dir}/assets — '
                    f'RTC prev_chunk will be sent raw (no normalize, R1 bias present)')
        except Exception as e:
            self.get_logger().warn(f'norm_stats load failed: {e}; RTC prev_chunk will be sent raw')

    def _load_ws_policy(self):
        """Connect to external serve_policy.py via WebSocket OR SHM (per transport param).

        transport=ws (default, JAX legacy): WebsocketClientPolicy → TCP loopback + msgpack
        transport=shm (V1 path, opt-in):    ShmClient → POSIX shm + zero-copy image
                                            (-5-7ms cycle P95 vs WS, §7.8 C.4)
        """
        transport = str(self.get_parameter('transport').value).lower()
        if transport == 'shm':
            self._connect_shm_policy()
        else:
            host = self.get_parameter('host').value
            port = self.get_parameter('port').value
            self.get_logger().info(f'Connecting to WebSocket policy at {host}:{port}')
            from openpi_client import websocket_client_policy
            self.policy = websocket_client_policy.WebsocketClientPolicy(host, port)
            self.get_logger().info('WebSocket policy connected')

    def _connect_shm_policy(self):
        """Attach to POSIX shm regions (server side serve_policy_v1.py --transport shm)."""
        req_name = str(self.get_parameter('shm_req_name').value)
        resp_name = str(self.get_parameter('shm_resp_name').value)
        self.get_logger().info(f'Attaching to SHM policy: req=/dev/shm/{req_name}, resp=/dev/shm/{resp_name}')

        # 加 kai0/scripts 到 sys.path 才能 import shm_transport
        import sys as _sys
        kai0_scripts = os.path.join(_KAI0_ROOT, 'scripts')
        if kai0_scripts not in _sys.path:
            _sys.path.insert(0, kai0_scripts)
        from shm_transport import ShmClient

        class _RosLogger:
            def __init__(self, node): self._n = node
            def info(self, m): self._n.get_logger().info(m)
            def warn(self, m): self._n.get_logger().warn(m)
            def error(self, m): self._n.get_logger().error(m)

        self.policy = ShmClient(
            req_name=req_name, resp_name=resp_name,
            attach_timeout_sec=60.0, logger=_RosLogger(self))
        self.get_logger().info('SHM policy attached')

    def _start_ws_server(self):
        """In 'both' mode, also serve the loaded policy via WebSocket."""
        ws_port = self.get_parameter('ws_port').value
        self.get_logger().info(f'Starting WebSocket server on :{ws_port}')

        from openpi.serving import websocket_policy_server
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=self.policy, host='0.0.0.0', port=ws_port,
            metadata=getattr(self.policy, 'metadata', {}))
        ws_thread = threading.Thread(target=server.serve_forever, daemon=True)
        ws_thread.start()
        self.get_logger().info(f'WebSocket server running on :{ws_port}')

    # ── Sensor callbacks (原版 deque 模式, maxlen=200) ──────────────

    def _rr_log_with_time(self, entity, data, stamp_sec):
        """Log to Rerun with explicit set_time + log, minimizing window for interleaving.

        NOTE: set_time + log is NOT atomic. This is safe because all callers
        (image/joint callbacks, FK timer) run on the ROS2 single-threaded executor.
        Do NOT switch to MultiThreadedExecutor without adding a Rerun lock.
        """
        rr = self._rr
        rr.set_time("ros_time", timestamp=stamp_sec)
        rr.log(entity, data)

    def _cb_img_front(self, msg):
        with self._sensor_lock:
            self._img_front_deque.append(msg)
        if self._use_rerun and self._rr_bridge is not None:
            try:
                img = self._rr_bridge.imgmsg_to_cv2(msg, 'rgb8')
                self._rr_log_with_time("images/top_head", self._rr.Image(img),
                                       _stamp_to_sec(msg.header.stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun img_front error: {e}')

    def _cb_img_left(self, msg):
        with self._sensor_lock:
            self._img_left_deque.append(msg)
        if self._use_rerun and self._rr_bridge is not None:
            try:
                img = self._rr_bridge.imgmsg_to_cv2(msg, 'rgb8')
                self._rr_log_with_time("images/hand_left", self._rr.Image(img),
                                       _stamp_to_sec(msg.header.stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun img_left error: {e}')

    def _cb_img_right(self, msg):
        with self._sensor_lock:
            self._img_right_deque.append(msg)
        if self._use_rerun and self._rr_bridge is not None:
            try:
                img = self._rr_bridge.imgmsg_to_cv2(msg, 'rgb8')
                self._rr_log_with_time("images/hand_right", self._rr.Image(img),
                                       _stamp_to_sec(msg.header.stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun img_right error: {e}')

    def _cb_depth_front(self, msg):
        """Project aligned depth from head camera into a 3D point cloud for Rerun."""
        if not self._use_rerun or self._rr is None:
            return
        if self._T_world_camF is None:
            return
        try:
            rr = self._rr
            bridge = self._rr_bridge or CvBridge()
            depth = bridge.imgmsg_to_cv2(msg, 'passthrough')  # uint16, mm
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) * 0.001
            else:
                depth_m = depth.astype(np.float32)

            # Downsample for performance (every 4th pixel)
            step = 4
            h, w = depth_m.shape
            v, u = np.mgrid[0:h:step, 0:w:step]
            z = depth_m[0:h:step, 0:w:step]
            valid = (z > 0.05) & (z < 1.5)
            u = u[valid].astype(np.float32)
            v = v[valid].astype(np.float32)
            z = z[valid]

            if len(z) == 0:
                return

            # Unproject using head camera intrinsics
            fx = self._cam_f_fx
            fy = self._cam_f_fy
            cx = self._cam_f_cx
            cy = self._cam_f_cy
            x_cam = (u - cx) * z / fx
            y_cam = (v - cy) * z / fy
            pts_cam = np.stack([x_cam, y_cam, z], axis=-1)  # (N, 3)

            # Transform to world frame
            R_wc = self._T_world_camF[:3, :3]
            t_wc = self._T_world_camF[:3, 3]
            pts_world = (R_wc @ pts_cam.T).T + t_wc

            stamp_sec = _stamp_to_sec(msg.header.stamp)
            rr.set_time("ros_time", timestamp=stamp_sec)
            rr.log("world/point_cloud", rr.Points3D(
                pts_world, radii=[0.002], colors=[[180, 180, 180]]))
        except Exception as e:
            self.get_logger().debug(f'Rerun depth_front error: {e}')

    def _cb_joint_left(self, msg):
        with self._sensor_lock:
            self._joint_left_deque.append(msg)
        if self._use_rerun and self._rr is not None:
            try:
                q = np.array(msg.position)
                self._rr_frame_idx_left += 1
                frame_idx = self._rr_frame_idx_left
                stamp = _stamp_to_sec(msg.header.stamp)
                rr = self._rr
                rr.set_time("ros_time", timestamp=stamp)
                rr.set_time("frame_left", sequence=frame_idx)
                for i in range(min(7, len(q))):
                    rr.log(f"timeseries/left_j{i}", rr.Scalars([float(q[i])]))
                # Queue FK work for the visualization timer (avoid blocking callback)
                if self._fk is not None and len(q) >= 7 and frame_idx % self._rr_fk_skip == 0:
                    self._rr_fk_queue.append(('left', q[:7].copy(), self._T_world_baseL, stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun left joint error: {e}')

    def _cb_joint_right(self, msg):
        with self._sensor_lock:
            self._joint_right_deque.append(msg)
        if self._use_rerun and self._rr is not None:
            try:
                q = np.array(msg.position)
                self._rr_frame_idx_right += 1
                frame_idx = self._rr_frame_idx_right
                stamp = _stamp_to_sec(msg.header.stamp)
                rr = self._rr
                rr.set_time("ros_time", timestamp=stamp)
                rr.set_time("frame_right", sequence=frame_idx)
                for i in range(min(7, len(q))):
                    rr.log(f"timeseries/right_j{i}", rr.Scalars([float(q[i])]))
                if self._fk is not None and len(q) >= 7 and frame_idx % self._rr_fk_skip == 0:
                    self._rr_fk_queue.append(('right', q[:7].copy(), self._T_world_baseR, stamp))
            except Exception as e:
                self.get_logger().debug(f'Rerun right joint error: {e}')

    # ── Execution control ────────────────────────────────────────

    def _cb_execute(self, msg):
        """ROS2 topic callback for /policy/execute (Bool)."""
        with self._exec_lock:
            was_enabled = self._execution_enabled
            if msg.data and not was_enabled:
                # Flush BEFORE enabling to prevent publishing stale actions.
                # EXCEPT in replay mode: stream_buffer was just preloaded with the
                # entire episode by _enter_replay_mode; flushing would discard it.
                if self._replay_mode != 'replay':
                    self._flush_stale_buffer()
            self._execution_enabled = msg.data
        mode_str = 'EXECUTE' if msg.data else 'OBSERVE'
        self.get_logger().info(f'[{mode_str}] 已切换到{"执行" if msg.data else "观测"}模式')
        if self._use_rerun:
            self._rr_queue_execute_mode(msg.data)

    def _toggle_execute(self):
        """Toggle execution mode."""
        with self._exec_lock:
            enabled = not self._execution_enabled
            if enabled:
                # Flush BEFORE enabling — see _cb_execute for replay-mode skip rationale.
                if self._replay_mode != 'replay':
                    self._flush_stale_buffer()
            self._execution_enabled = enabled
        mode_str = 'EXECUTE' if enabled else 'OBSERVE'
        self.get_logger().info(f'[{mode_str}] 已切换到{"执行" if enabled else "观测"}模式')
        if self._use_rerun:
            self._rr_queue_execute_mode(enabled)

    def _switch_to_observe(self):
        """Switch to observe mode if currently executing."""
        with self._exec_lock:
            if self._execution_enabled:
                self._execution_enabled = False
                self.get_logger().info('[OBSERVE] 已切换到观测模式')
                if self._use_rerun:
                    self._rr_queue_execute_mode(False)

    def _rr_queue_execute_mode(self, enabled):
        """Thread-safe: queue a Rerun execute_mode scalar for the executor timer to log.

        Safe to call from any thread (keyboard listener, ROS2 executor, etc.).
        """
        stamp_sec = self.get_clock().now().nanoseconds * 1e-9
        val = 1.0 if enabled else 0.0
        # Capture val in closure to avoid late-binding issues
        self._rr_event_queue.append((
            "timeseries/execute_mode",
            lambda v=val: self._rr.Scalars([v]),
            stamp_sec,
        ))

    def _flush_stale_buffer(self):
        """Discard stale actions, seed last_action with current joint pos for smooth transition.

        May be called with or without _exec_lock held (from _cb_execute/_toggle_execute
        under _exec_lock, or from _publish_action without it).

        Lock ordering: _exec_lock → _sensor_lock. Never acquire _exec_lock while
        holding _sensor_lock to avoid deadlock."""
        with self._sensor_lock:
            jl = self._joint_left_deque[-1] if self._joint_left_deque else None
            jr = self._joint_right_deque[-1] if self._joint_right_deque else None

        seed = None
        if jl is not None and jr is not None:
            ql = np.array(jl.position[:7])
            qr = np.array(jr.position[:7])
            if len(ql) < 7:
                ql = np.pad(ql, (0, 7 - len(ql)))
            if len(qr) < 7:
                qr = np.pad(qr, (0, 7 - len(qr)))
            seed = np.concatenate([ql, qr])  # always 14-dim
        self.stream_buffer.flush(seed_action=seed)
        # Clear RTC prev_chunk so next inference starts fresh (no guidance toward
        # a stale trajectory). Next call falls back to base_step (pi0_rtc.py:351).
        with self._rtc_lock:
            self._rtc_prev_chunk = None
        # Exp-1a: reset obs-state low-pass so the first post-observe frame seeds
        # from a fresh state (not smoothed against observe-period state).
        self._obs_state_lp_prev = None

    def _keyboard_listener(self):
        """Non-blocking keyboard listener for interactive control.

        Enter/Space → toggle execute/observe
        q/Esc       → switch to observe mode
        """
        import termios
        import tty

        self.get_logger().info(
            'Keyboard control: [Enter/Space] toggle execute | [q/Esc] → observe')
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
        except (termios.error, ValueError, OSError):
            # Not a real terminal (e.g., launched via ros2 launch)
            # Fall back to line-buffered stdin
            self._keyboard_listener_line_mode()
            return

        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ('\n', '\r', ' '):
                        self._toggle_execute()
                    elif ch == '\x1b':
                        # Drain trailing escape sequence bytes (e.g. arrow keys: \x1b[A)
                        # to avoid interpreting them as separate keypresses
                        has_seq = False
                        while select.select([sys.stdin], [], [], 0.05)[0]:
                            sys.stdin.read(1)
                            has_seq = True
                        if has_seq:
                            # Was a multi-byte escape sequence (arrow key etc.), ignore
                            continue
                        # Bare Esc key — fall through to observe mode
                        self._switch_to_observe()
                    elif ch == 'q':
                        self._switch_to_observe()
        except Exception as e:
            self.get_logger().warn(f'Keyboard listener exited: {e}')
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _keyboard_listener_line_mode(self):
        """Fallback: line-buffered stdin (for ros2 launch environments)."""
        try:
            # Check stdin is readable (may be closed/redirected under ros2 launch)
            if sys.stdin is None or sys.stdin.closed:
                self.get_logger().info('stdin not available, keyboard control disabled')
                return
            sys.stdin.fileno()  # will raise if stdin is not a real fd
        except (ValueError, OSError):
            self.get_logger().info('stdin not available, keyboard control disabled')
            return

        while rclpy.ok():
            try:
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    line = sys.stdin.readline()
                    if not line:
                        # EOF — stdin closed
                        self.get_logger().info('stdin EOF, keyboard control stopped')
                        return
                    line = line.strip().lower()
                    if line in ('', 'toggle', 't'):
                        self._toggle_execute()
                    elif line in ('q', 'quit', 'stop'):
                        self._switch_to_observe()
            except (OSError, ValueError):
                # stdin closed/broken
                self.get_logger().info('stdin error, keyboard control stopped')
                return
            except Exception:
                pass

    # ── Rerun visualization ────────────────────────────────────────

    def _rr_process_fk_queue(self):
        """Timer callback: drain FK queue, event queue, and predicted traj.

        Time-budgeted to avoid blocking the executor for too long (target <30ms).
        """
        rr = self._rr
        if rr is None:
            return

        t_budget_start = time.monotonic()
        BUDGET_SEC = 0.030  # 30ms max per timer tick

        # 1. Drain cross-thread event queue (keyboard toggle, etc.) — always process first (cheap)
        while self._rr_event_queue:
            try:
                entity, data_factory, stamp_sec = self._rr_event_queue.popleft()
            except IndexError:
                break
            try:
                rr.set_time("ros_time", timestamp=stamp_sec)
                rr.log(entity, data_factory())
            except Exception as e:
                self.get_logger().debug(f'Rerun event queue error: {e}')

        if self._fk is None:
            return

        # 2. Adaptive skip for arm FK queue
        queue_len = len(self._rr_fk_queue)
        if queue_len >= self._rr_fk_queue.maxlen:
            self._rr_fk_drops += 1
            if self._rr_fk_skip < 15:
                self._rr_fk_skip = min(self._rr_fk_skip + 1, 15)
            if self._rr_fk_drops % 50 == 1:
                self.get_logger().warn(
                    f'Rerun FK queue full ({queue_len}), increased skip to {self._rr_fk_skip} '
                    f'(total drops: {self._rr_fk_drops})')
        elif queue_len <= self._rr_fk_queue.maxlen // 4 and self._rr_fk_skip > 3:
            # Queue has headroom, gradually restore skip
            self._rr_fk_skip = max(self._rr_fk_skip - 1, 3)

        # 3. Drain arm FK requests (with time budget)
        processed = 0
        while self._rr_fk_queue and processed < 10:
            if time.monotonic() - t_budget_start > BUDGET_SEC:
                break
            try:
                arm_label, q6, T_base, stamp = self._rr_fk_queue.popleft()
            except IndexError:
                break
            processed += 1
            try:
                rr.set_time("ros_time", timestamp=stamp)
                ee_pos = self._rr_log_arm_fk(arm_label, q6, T_base)
                if ee_pos is not None:
                    if arm_label == 'left':
                        self._actual_trail_left.append(ee_pos.copy())
                        trail = np.array(list(self._actual_trail_left))
                        rr.log("world/actual/left_trail", rr.Points3D(
                            trail, colors=[[50, 100, 255]], radii=[0.002]))
                    else:
                        self._actual_trail_right.append(ee_pos.copy())
                        trail = np.array(list(self._actual_trail_right))
                        rr.log("world/actual/right_trail", rr.Points3D(
                            trail, colors=[[50, 255, 100]], radii=[0.002]))
            except Exception as e:
                self.get_logger().debug(f'Rerun FK viz error ({arm_label}): {e}')

        # 4. Process predicted trajectory FK (time-budgeted: only one per tick)
        if self._rr_pred_queue and (time.monotonic() - t_budget_start) < BUDGET_SEC:
            try:
                actions, stamp = self._rr_pred_queue.popleft()
            except IndexError:
                pass
            else:
                try:
                    rr.set_time("ros_time", timestamp=stamp)
                    self._rr_log_predicted_trajectory(actions)
                except Exception as e:
                    self.get_logger().debug(f'Rerun predicted traj error: {e}')
            # Log if predictions are being dropped
            if self._rr_pred_queue:
                self._rr_pred_drops += len(self._rr_pred_queue)
                self._rr_pred_queue.clear()
                if self._rr_pred_drops % 10 == 1:
                    self.get_logger().info(
                        f'Rerun predicted traj queue overflow (total drops: {self._rr_pred_drops})')

    def _init_rerun(self):
        """Initialize Rerun viewer with blueprint and static elements."""
        try:
            import rerun as rr
            import rerun.blueprint as rrb
        except ImportError:
            self.get_logger().warn('rerun not installed, disabling visualization')
            self._use_rerun = False
            return

        self._rr = rr
        self._rr_bridge = CvBridge()

        # Rerun >=0.21 uses new_recording(); older versions use init()
        if hasattr(rr, 'new_recording'):
            try:
                self._rr_rec = rr.new_recording("inference_viz", make_default=True, make_thread_default=True)
            except TypeError:
                # Older new_recording() may not support make_thread_default
                self._rr_rec = rr.new_recording("inference_viz", make_default=True)
        else:
            self._rr_rec = rr.init("inference_viz")
            # Old Rerun: ensure recording is set as default for all threads
            if self._rr_rec is not None and hasattr(self._rr_rec, 'set_global'):
                self._rr_rec.set_global()

        # Spawn viewer or save to file
        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            try:
                rr.spawn()
            except Exception as e:
                from datetime import datetime
                rrd_path = f'/tmp/inference_viz_{datetime.now():%Y%m%d_%H%M%S}.rrd'
                self.get_logger().warn(
                    f'rr.spawn() failed ({e}), saving to {rrd_path}. '
                    f'View with: rerun {rrd_path}')
                rr.save(rrd_path)
        else:
            from datetime import datetime
            rrd_path = f'/tmp/inference_viz_{datetime.now():%Y%m%d_%H%M%S}.rrd'
            self.get_logger().warn(
                f'No display detected, Rerun saving to {rrd_path}. '
                f'View with: rerun {rrd_path}')
            rr.save(rrd_path)
        # Pre-log empty images so Rerun blueprint can resolve entity paths
        import numpy as np
        _placeholder = rr.Image(np.zeros((1, 1, 3), dtype=np.uint8))
        for _entity in ("images/top_head", "images/hand_left", "images/hand_right"):
            rr.log(_entity, _placeholder, static=True)

        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", contents="world/**", name="3D Scene"),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="images/top_head", name="Head"),
                        rrb.Spatial2DView(origin="images/hand_left", name="Left"),
                        rrb.Spatial2DView(origin="images/hand_right", name="Right"),
                    ),
                    rrb.TimeSeriesView(origin="timeseries", contents="timeseries/**",
                                       name="Telemetry"),
                    row_shares=[1, 1],
                ),
                column_shares=[3, 2],
            ),
        ))

        # Load calibration for FK
        calib_path = self.get_parameter('calibration_config').value
        if not calib_path or not os.path.isfile(calib_path):
            self.get_logger().warn(
                f'calibration_config not found: {calib_path!r}, '
                'FK visualization disabled')
            self._fk = None
        else:
            self._load_calibration_and_fk(calib_path)

        # Log initial execute mode
        rr.log("timeseries/execute_mode",
               rr.Scalars([1.0 if self._execution_enabled else 0.0]))

        self.get_logger().info('Rerun viewer initialized')

    def _load_calibration_and_fk(self, calib_path):
        """Load calibration YAML and FK, log static mesh elements."""
        rr = self._rr

        # Load calibration
        with open(calib_path) as f:
            calib = yaml.safe_load(f)
        transforms = calib['transforms']
        for key in transforms:
            transforms[key] = np.array(transforms[key])

        self._T_world_baseL = transforms['T_world_baseL']
        self._T_world_baseR = transforms['T_world_baseR']
        self._T_world_camF = transforms.get('T_world_camF')

        # Head camera intrinsics for depth → point cloud projection
        intrinsics = calib.get('intrinsics', {}).get('cam_f', {})
        self._cam_f_fx = intrinsics.get('fx', 606.5)
        self._cam_f_fy = intrinsics.get('fy', 605.7)
        self._cam_f_cx = intrinsics.get('cx', 326.5)
        self._cam_f_cy = intrinsics.get('cy', 256.9)

        # FK — find calib/ dir relative to project root or calibration_config location
        calib_dir = None
        for candidate in [
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', '..', '..', '..', 'calib')),
            os.path.normpath(os.path.join(_KAI0_ROOT, '..', 'calib')),
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(calib_path)),
                                          '..', 'calib')),
        ]:
            if os.path.isfile(os.path.join(candidate, 'piper_fk.py')):
                calib_dir = candidate
                break
        if calib_dir is None:
            self.get_logger().warn('Cannot find calib/piper_fk.py, FK disabled')
            self._fk = None
            return
        if calib_dir not in sys.path:
            sys.path.insert(0, calib_dir)
        from piper_fk import PiperFK
        self._fk = PiperFK()

        # Base markers
        for label, T_key in [('baseL', 'T_world_baseL'), ('baseR', 'T_world_baseR')]:
            pos = transforms[T_key][:3, 3]
            rr.log(f"world/{label}", rr.Points3D(
                [pos], colors=[[255, 255, 0]], radii=[0.015], labels=[label]),
                static=True)

        # Head camera frustum
        if 'T_world_camF' in transforms:
            T_head = transforms['T_world_camF']
            head_pos = T_head[:3, 3]
            R_head = T_head[:3, :3]
            d, hw_, hh_ = 0.15, 0.09, 0.07
            corners = [[-hw_, -hh_, d], [hw_, -hh_, d], [hw_, hh_, d], [-hw_, hh_, d]]
            o = head_pos.tolist()
            cs = [(R_head @ np.array(c) + head_pos).tolist() for c in corners]
            lines = [[o, cs[0]], [o, cs[1]], [o, cs[2]], [o, cs[3]],
                     [cs[0], cs[1]], [cs[1], cs[2]], [cs[2], cs[3]], [cs[3], cs[0]]]
            rr.log("world/head_cam_frustum", rr.LineStrips3D(
                lines, colors=[[255, 50, 50]], radii=[0.003]), static=True)

        # Workspace box — centered between the two arm bases, offset in the average
        # forward direction (local Y of each base frame) so it matches actual calibration.
        base_mid = (self._T_world_baseL[:3, 3] + self._T_world_baseR[:3, 3]) / 2.0
        fwd_L = self._T_world_baseL[:3, 1]  # local Y axis of left base in world
        fwd_R = self._T_world_baseR[:3, 1]  # local Y axis of right base in world
        fwd_avg = (fwd_L + fwd_R) / 2.0
        fwd_norm = np.linalg.norm(fwd_avg)
        if fwd_norm > 1e-6:
            fwd_avg /= fwd_norm
        else:
            fwd_avg = np.array([0.0, 1.0, 0.0])
        ws_center = base_mid + fwd_avg * 0.30 + np.array([0.0, 0.0, 0.05])
        base_span = np.linalg.norm(self._T_world_baseL[:3, 3] - self._T_world_baseR[:3, 3])
        ws_width = max(1.0, base_span + 0.6)   # wider to cover full arm reach
        rr.log("world/workspace", rr.Boxes3D(
            centers=[ws_center.tolist()], sizes=[[ws_width, 0.9, 0.7]],
            colors=[[255, 255, 255, 40]]), static=True)

        # Load arm meshes — search multiple candidate paths (source layout, install layout, KAI0_ROOT)
        _mesh_suffix = os.path.join('train_deploy_alignment', 'inference', 'agilex',
                                     'Piper_ros_private-ros-noetic', 'src',
                                     'piper_description', 'meshes')
        mesh_dir = None
        for _mesh_base in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', '..', 'kai0'),       # source layout
            _KAI0_ROOT,                                           # env / auto-detected
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', '..', '..', '..', 'kai0'),  # install layout
        ]:
            candidate = os.path.normpath(os.path.join(_mesh_base, _mesh_suffix))
            if os.path.isdir(candidate):
                mesh_dir = candidate
                break
        if mesh_dir is None:
            mesh_dir = os.path.normpath(os.path.join(_KAI0_ROOT, _mesh_suffix))
        mesh_names = ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5',
                      'link6', 'gripper_base', 'link7', 'link8']
        self._arm_mesh_loaded = False
        if os.path.isdir(mesh_dir):
            try:
                import trimesh
                for arm_label in ('left', 'right'):
                    arm_color = [100, 100, 255] if arm_label == 'left' else [100, 255, 100]
                    for mesh_name in mesh_names:
                        stl_path = os.path.join(mesh_dir, f'{mesh_name}.STL')
                        if os.path.exists(stl_path):
                            m = trimesh.load(stl_path)
                            rr.log(f"world/{arm_label}/{mesh_name}", rr.Mesh3D(
                                vertex_positions=m.vertices,
                                triangle_indices=m.faces,
                                vertex_colors=np.full((len(m.vertices), 3),
                                                      arm_color, dtype=np.uint8),
                            ), static=True)
                self._arm_mesh_loaded = True
                self.get_logger().info(f'Loaded arm meshes from {mesh_dir}')
            except ImportError:
                self.get_logger().warn('trimesh not available, skipping arm mesh')
        else:
            self.get_logger().warn(f'Mesh dir not found: {mesh_dir}')

    def _rr_log_arm_fk(self, arm_label, q7, T_world_base):
        """Update arm link transforms via FK (drives mesh positions).

        Args:
            q7: 7-element array [j0..j5, gripper_open]. j0-j5 for FK, gripper_open for finger viz.
        """
        if self._fk is None or self._rr is None:
            return
        rr = self._rr

        q6_rad = q7[:6]
        gripper_val = float(q7[6]) if len(q7) > 6 else 0.0

        link_Ts = self._fk.fk_all_links(q6_rad)
        T_base = np.array(T_world_base)

        # Base link
        rr.log(f"world/{arm_label}/base_link", rr.Transform3D(
            translation=T_base[:3, 3], mat3x3=T_base[:3, :3]))

        # Links 1-6
        for idx, T_link in enumerate(link_Ts):
            T_world_link = T_base @ T_link
            rr.log(f"world/{arm_label}/link{idx+1}", rr.Transform3D(
                translation=T_world_link[:3, 3], mat3x3=T_world_link[:3, :3]))

        # Gripper base = link6 pose
        T_world_link6 = T_base @ link_Ts[5]
        rr.log(f"world/{arm_label}/gripper_base", rr.Transform3D(
            translation=T_world_link6[:3, 3], mat3x3=T_world_link6[:3, :3]))

        # Gripper fingers (link7, link8) — driven by gripper_val (q[6])
        # gripper_val ~0 = closed, ~0.04 = open. Map to lateral finger offset.
        finger_spread = np.clip(gripper_val, 0.0, 0.04) / 0.04  # normalize to [0, 1]
        max_lateral = 0.025  # max lateral offset per finger (m)
        for link_name, rpy, sign in [('link7', [np.pi/2, 0, 0], 1.0),
                                      ('link8', [np.pi/2, 0, -np.pi], -1.0)]:
            T_offset = np.eye(4)
            T_offset[:3, :3] = R_.from_euler('xyz', rpy).as_matrix()
            T_offset[:3, 3] = [sign * finger_spread * max_lateral, 0, 0.1358]
            T_w = T_world_link6 @ T_offset
            rr.log(f"world/{arm_label}/{link_name}", rr.Transform3D(
                translation=T_w[:3, 3], mat3x3=T_w[:3, :3]))

        # EE marker (link6 = end-effector, already computed above)
        ee_pos = T_world_link6[:3, 3]
        ee_color = [50, 100, 255] if arm_label == 'left' else [50, 255, 100]
        rr.log(f"world/{arm_label}/ee", rr.Points3D(
            [ee_pos], colors=[ee_color], radii=[0.008]))

        return ee_pos

    # Piper joint limits (rad) — used to validate FK inputs before visualization
    _JOINT_LIMITS_RAD = np.array([2.618, 1.571, 1.571, 1.745, 1.571, 2.618])  # ~[150,90,90,100,90,150]°

    def _rr_log_predicted_trajectory(self, actions):
        """Convert action chunk [50, 14] to EE paths via FK and log as LineStrips3D."""
        if self._fk is None or self._rr is None:
            return
        rr = self._rr

        limits = self._JOINT_LIMITS_RAD
        left_path, right_path = [], []
        for t in range(actions.shape[0]):
            q_l = actions[t, 0:6]
            q_r = actions[t, 7:13]
            # Skip steps with out-of-range joints to prevent wild FK output
            if np.any(np.abs(q_l) > limits * 1.5) or np.any(np.abs(q_r) > limits * 1.5):
                continue
            T_l = self._T_world_baseL @ self._fk.fk_homogeneous(q_l)
            T_r = self._T_world_baseR @ self._fk.fk_homogeneous(q_r)
            left_path.append(T_l[:3, 3])
            right_path.append(T_r[:3, 3])

        if len(left_path) < 2:
            return

        left_arr = np.array(left_path)
        right_arr = np.array(right_path)

        rr.log("world/predicted/left_traj", rr.LineStrips3D(
            [left_arr], colors=[[50, 100, 255]], radii=[0.003]))
        rr.log("world/predicted/right_traj", rr.LineStrips3D(
            [right_arr], colors=[[50, 255, 100]], radii=[0.003]))

        # Start/end markers
        rr.log("world/predicted/left_endpoints", rr.Points3D(
            [left_arr[0], left_arr[-1]], colors=[[50, 100, 255]],
            radii=[0.008], labels=["start", "end"]))
        rr.log("world/predicted/right_endpoints", rr.Points3D(
            [right_arr[0], right_arr[-1]], colors=[[50, 255, 100]],
            radii=[0.008], labels=["start", "end"]))

    # ── Frame sync (复刻原版 get_frame, 基于 min(timestamp) 对齐) ──

    def _get_synced_frame(self):
        """Return timestamp-aligned (img_front, img_left, img_right, joint_left, joint_right)
        or None if any sensor data is missing/stale."""
        with self._sensor_lock:
            if (len(self._img_front_deque) == 0
                    or len(self._img_left_deque) == 0
                    or len(self._img_right_deque) == 0
                    or len(self._joint_left_deque) == 0
                    or len(self._joint_right_deque) == 0):
                return None

            # Sync time = min of latest timestamps across 3 cameras
            frame_time = min(
                _stamp_to_sec(self._img_front_deque[-1].header.stamp),
                _stamp_to_sec(self._img_left_deque[-1].header.stamp),
                _stamp_to_sec(self._img_right_deque[-1].header.stamp),
            )

            # Check all sensors have data at or after frame_time
            for dq, name in [
                (self._img_front_deque, 'img_front'),
                (self._img_left_deque, 'img_left'),
                (self._img_right_deque, 'img_right'),
                (self._joint_left_deque, 'joint_left'),
                (self._joint_right_deque, 'joint_right'),
            ]:
                if len(dq) == 0 or _stamp_to_sec(dq[-1].header.stamp) < frame_time:
                    return None

            # Pop frames up to frame_time (discard stale data)
            def _pop_synced(dq):
                while len(dq) > 0 and _stamp_to_sec(dq[0].header.stamp) < frame_time:
                    dq.popleft()
                if len(dq) == 0:
                    return None
                return dq.popleft()

            img_front_msg = _pop_synced(self._img_front_deque)
            img_left_msg = _pop_synced(self._img_left_deque)
            img_right_msg = _pop_synced(self._img_right_deque)
            joint_left_msg = _pop_synced(self._joint_left_deque)
            joint_right_msg = _pop_synced(self._joint_right_deque)

            if any(m is None for m in (img_front_msg, img_left_msg, img_right_msg,
                                       joint_left_msg, joint_right_msg)):
                return None

        # 强制输出 BGR: RealSense ROS2 默认 rgb8, cv_bridge 会自动转换
        # 后续管线 jpeg_mapping + COLOR_BGR2RGB 假设输入为 BGR
        if self._fast_obs_pipeline:
            # P2 Step 2: 跳 CvBridge, 直接 np.frombuffer view msg.data.
            # multi_camera_node publish 时 encoding='rgb8' (line 318) — 这里得到的 numpy
            # 直接就是 RGB8 (跟模型期望一致, 不需要后续 cvtColor).
            img_front = np.frombuffer(img_front_msg.data, dtype=np.uint8).reshape(
                img_front_msg.height, img_front_msg.width, 3)
            img_left = np.frombuffer(img_left_msg.data, dtype=np.uint8).reshape(
                img_left_msg.height, img_left_msg.width, 3)
            img_right = np.frombuffer(img_right_msg.data, dtype=np.uint8).reshape(
                img_right_msg.height, img_right_msg.width, 3)
        else:
            img_front = self.bridge.imgmsg_to_cv2(img_front_msg, 'bgr8')
            img_left = self.bridge.imgmsg_to_cv2(img_left_msg, 'bgr8')
            img_right = self.bridge.imgmsg_to_cv2(img_right_msg, 'bgr8')

        # C.2 2026-05-23: 捕获 head 相机帧 stamp (ns), 让 _record_latency_sample 准确测
        # image_age. A.2 pipelined mode 下 worker pop deque 导致 main 读 deque[-1] 偶发空
        # (70% NaN); 改成沿着 obs 通过 tuple 把"这次推理实际用的帧"的 stamp 传过去, 测量稳定.
        head_stamp = img_front_msg.header.stamp
        head_stamp_ns = head_stamp.sec * 1_000_000_000 + head_stamp.nanosec

        return img_front, img_left, img_right, joint_left_msg, joint_right_msg, head_stamp_ns

    # ── Image preprocessing (严格复刻原版管线) ──────────────────────

    @staticmethod
    def _jpeg_mapping(img):
        """JPEG encode/decode 对齐训练数据的 MP4 视频压缩 artifacts.

        原版链路: passthrough → imencode (按 BGR 编码) → imdecode (固定输出 BGR)
        无论输入是 rgb8 还是 bgr8, 输出均为 BGR (OpenCV 约定).
        """
        img = cv2.imencode(".jpg", img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img

    def _get_observation(self):
        """Pack current sensor data into policy input. Returns obs dict (or None).

        Thin wrapper around `_get_observation_with_stamp` 丢掉 stamp — 保留这个签名
        向后兼容任何"只关心 obs"的调用方. 新代码 (worker / inference loop main 路径)
        应直接用 _get_observation_with_stamp() 拿 stamp.
        """
        obs, _stamp = self._get_observation_with_stamp()
        return obs

    def _get_observation_with_stamp(self):
        """Pack current sensor data into policy input + return head 帧 stamp.

        Returns (obs_dict, head_stamp_ns) or (None, None) if sensor data 不全.
        head_stamp_ns 用于准确测 image_age (尤其 A.2 pipelined mode 下 worker 已经
        pop 走 deque, main 读 deque[-1] 偶发空). stamp 不进 obs dict, 走 tuple 侧通道
        避免 JAX/V1 transform 链兼容性风险.

        严格复刻原版 update_observation_window + inference_fn 的图像管线:
        1. get_synced_frame (timestamp 对齐)
        2. JPEG encode/decode (训练对齐, P2 fast_obs_pipeline 可跳过)
        3. BGR → RGB (P2 fast_obs_pipeline 可跳过)
        4. resize_with_pad(224, 224) — 保持宽高比, 零填充
        5. HWC → CHW
        """
        frame = self._get_synced_frame()
        if frame is None:
            return None, None

        img_front, img_left, img_right, joint_left_msg, joint_right_msg, head_stamp_ns = frame

        from openpi_client import image_tools

        # 原版顺序: front, right, left (camera_names = [front, right, left])
        if self._fast_obs_pipeline:
            # P2 Step 1: 跳 JPEG mapping + cvtColor. img_* 已是 RGB8 numpy view from
            # _get_synced_frame 的 np.frombuffer path. 直接进 resize.
            imgs = [img_front, img_right, img_left]
        else:
            imgs = [
                self._jpeg_mapping(img_front),
                self._jpeg_mapping(img_right),
                self._jpeg_mapping(img_left),
            ]
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
        # 与原版一致: 相同分辨率时走 batch resize, 否则逐张
        _ih, _iw = self._obs_image_h, self._obs_image_w
        if imgs[0].shape == imgs[1].shape == imgs[2].shape:
            imgs = list(image_tools.resize_with_pad(np.array(imgs), _ih, _iw))
        else:
            imgs = [image_tools.resize_with_pad(im[np.newaxis], _ih, _iw)[0] for im in imgs]

        qpos = np.concatenate((
            np.array(joint_left_msg.position),
            np.array(joint_right_msg.position),
        ), axis=0)

        # Exp-1a: low-pass the state fed to the model to break the closed-loop
        # positive feedback (state jitter → action jitter → state jitter → ~1Hz
        # limit cycle). β=1.0 → no-op (bit-identical legacy). Gripper dims (6,13)
        # bypass smoothing (binary; EMA traps half-grasp).
        if self._obs_state_lp_alpha < 1.0:
            b = self._obs_state_lp_alpha
            if self._obs_state_lp_prev is None or self._obs_state_lp_prev.shape != qpos.shape:
                self._obs_state_lp_prev = qpos.copy()
            sm = b * qpos + (1.0 - b) * self._obs_state_lp_prev
            if qpos.shape[0] >= 14:
                sm[6] = qpos[6]; sm[13] = qpos[13]  # keep raw gripper
            self._obs_state_lp_prev = sm.copy()
            qpos = sm

        # RTC 式 commanded-proprio: execute 时用【上次下发的关节命令】当 proprio, 而非实测。
        # 去传感噪声/臂滞后 → 连续推理下 chunk 一致 (上游 SoftFold 平滑机制; 喂"刚下发"而非
        # "chunk 末步" → 连续模式正确时序、不前瞻、随臂推进不卡死)。images 仍用实测。
        # 首帧 (last_published=None) 或 observe 时回退实测。EE-mode 下 server 由此 14D 关节算 EE6D proprio。
        # 注: ③ firmware 模式下 _last_published_action 是 EE-cmd (非关节), 不能当 joint proprio →
        # 该模式回退实测关节 (固件平滑跟随, 实测本就干净)。仅 joint-IK 模式用 commanded-proprio。
        if (self._proprio_cmd_fb and self._execution_enabled
                and self._ee_ctrl != 'firmware'
                and self._last_published_action is not None):
            lp = np.asarray(self._last_published_action, dtype=float)
            if lp.shape[0] >= 14:
                qpos = lp[:14].copy()

        obs = {
            'state': qpos,
            'images': {
                'top_head':   imgs[0].transpose(2, 0, 1),   # CHW
                'hand_right': imgs[1].transpose(2, 0, 1),
                'hand_left':  imgs[2].transpose(2, 0, 1),
            },
            'prompt': self.prompt,
        }
        # Domain-conditioned models (action_head_cond_num_domains>0): inject fixed dataset_id so
        # Pi0.embed_suffix prepends the per-domain token (vis=1). Passthrough via agilex_policy →
        # Observation.dataset_id. -1 keeps obs bit-identical to plain pi05 (no extra key).
        if self._dataset_id >= 0:
            obs['dataset_id'] = np.int32(self._dataset_id)
        # C.2: stamp 不进 obs dict (避免 JAX/V1 transform 链兼容性风险),
        # 走 tuple 侧通道. obs 完全 bit-identical 旧版.
        return obs, int(head_stamp_ns)

    # ── Inference loop ──────────────────────────────────────────────

    # ── B1 latency profile helpers ──────────────────────────────────

    def _record_latency_sample(
        self, t_loop_start, t_obs_start, t_obs_end,
        t_ws_send, t_ws_recv, t_buffer_done,
        obs, result,
        obs_stamp_ns=None,  # C.2: head 帧 stamp 侧通道, 优先于 deque[-1] fallback
    ):
        """Write one CSV row capturing 11-segment client-side latency profile.

        Segments (ms):
            t_image_age           : C.2 优先 obs_stamp_ns; fallback deque[-1] stamp
            t_obs_construct       : t_obs_end - t_obs_start (_get_observation cost)
            t_ws_full_rtt         : t_ws_recv - t_ws_send (raw infer() round-trip)
            t_ws_overhead         : t_ws_full_rtt - server_total (= serialize+transit)
            server_*              : from result["policy_timing"] (B1 server side)
            t_buffer_integrate    : push chunk to stream_buffer
            t_loop_total          : t_buffer_done - t_loop_start (incl all of above)
        """
        if self._lat_profile_fp is None:
            return
        try:
            self._lat_profile_cycle += 1
            now_ros = self.get_clock().now()
            img_age_ms = float('nan')
            # C.2 2026-05-23: 优先用调用方传的 obs_stamp_ns (本次推理实际用的 head 帧
            # stamp, 走 tuple 侧通道 — 不进 obs dict, 兼容 JAX/V1 transform 链).
            # A.2 pipelined mode 下 worker 已经 pop 走 deque, main 读 deque[-1] 偶发
            # 空 (70% NaN); 用 stamp 参数避免那个 race.
            if obs_stamp_ns is not None:
                img_age_ns = now_ros.nanoseconds - int(obs_stamp_ns)
                img_age_ms = img_age_ns / 1e6
            else:
                # Legacy fallback: 读 deque[-1] (pre-C.2 行为, JAX 路径 / 同步模式走这里)
                with self._sensor_lock:
                    if self._img_front_deque:
                        latest_img = self._img_front_deque[-1]
                        img_stamp = latest_img.header.stamp
                        img_age_ns = (now_ros.nanoseconds
                                      - (img_stamp.sec * 1_000_000_000 + img_stamp.nanosec))
                        img_age_ms = img_age_ns / 1e6

            pt = result.get('policy_timing', {}) if isinstance(result, dict) else {}
            obs_construct_ms = (t_obs_end - t_obs_start) * 1000
            ws_full_rtt_ms = (t_ws_recv - t_ws_send) * 1000
            server_total_ms = float(pt.get('total_ms', 0.0))
            ws_overhead_ms = ws_full_rtt_ms - server_total_ms
            buffer_integrate_ms = (t_buffer_done - t_ws_recv) * 1000
            loop_total_ms = (t_buffer_done - t_loop_start) * 1000

            self._lat_profile_fp.write(
                f"{self._lat_profile_cycle},"
                f"{img_age_ms:.3f},{obs_construct_ms:.3f},"
                f"{ws_full_rtt_ms:.3f},{ws_overhead_ms:.3f},"
                f"{float(pt.get('preproc_ms', 0)):.3f},"
                f"{float(pt.get('state_encode_ms', 0)):.3f},"
                f"{float(pt.get('infer_ms', 0)):.3f},"
                f"{float(pt.get('postproc_ms', 0)):.3f},"
                f"{server_total_ms:.3f},"
                f"{buffer_integrate_ms:.3f},{loop_total_ms:.3f}\n"
            )
        except Exception as e:
            # Profiling MUST NOT break production inference; swallow + log warn.
            self.get_logger().warn(f"latency profile write error: {e}")

    def _inference_loop(self):
        """Background thread: continuously infer and push to buffer."""
        # Wait for policy to be loaded
        while self.policy is None and rclpy.ok():
            time.sleep(0.1)

        # Set thread-local Rerun recording so set_time calls don't clash with callbacks.
        # If set_thread_local is unavailable (old Rerun), disable Rerun in this thread
        # to prevent timeline corruption between inference thread and executor thread.
        self._rr_infer_thread = False
        if self._use_rerun and self._rr is not None and self._rr_rec is not None:
            if hasattr(self._rr_rec, 'set_thread_local'):
                self._rr_rec.set_thread_local()
                self._rr_infer_thread = True
            else:
                self.get_logger().warn(
                    'Rerun RecordingStream.set_thread_local() not available, '
                    'disabling Rerun logging in inference thread to prevent timeline corruption')

        self.get_logger().info('Inference loop started')

        # Warmup
        self.get_logger().info('Waiting for sensor data...')
        while rclpy.ok():
            obs = self._get_observation()
            if obs is not None:
                break
            time.sleep(0.1)

        self.get_logger().info('Running warmup inference...')
        t0 = time.monotonic()
        try:
            self.policy.infer(obs)
        except Exception as e:
            self.get_logger().warn(f'Warmup failed: {e}')
        self.get_logger().info(f'Warmup done in {(time.monotonic()-t0)*1000:.0f}ms')

        # Main loop
        while rclpy.ok():
            t_start = time.monotonic()
            # Replay-mode short-circuit: stream_buffer was filled on mode-entry;
            # publish_timer drains at 30 Hz. Here we only track progress + handle
            # end-of-episode. Inference path is fully bypassed.
            if self._replay_mode == 'replay':
                try:
                    self._tick_replay()
                except Exception as e:
                    self.get_logger().error(f'replay tick error: {e}')
                time.sleep(0.1)  # 10 Hz progress check
                continue
            if self._replay_mode == 'idle':
                time.sleep(0.1)
                continue
            # ② 开环整 chunk: execute 时, 当前 chunk 没排到阈值就不重推。让臂跑完大部分
            # chunk → 重推时用新图像 (任务推进) + ① pred_proprio≈当前。observe 不 gate。
            # 阈值【自适应】: 预留 = 实测整周期(infer+IK+integrate)会排空的步数 + 余量,
            # 保证新 chunk 在 buffer 见底前到达 → 边界无停顿 (不必手调 min_remaining)。
            if self._open_loop_chunk and self._execution_enabled:
                reserve = int(np.ceil(self._last_cycle_s * float(self.publish_rate))) + 3
                thr = max(self._open_loop_min_remaining, reserve)
                with self.stream_buffer.lock:
                    remaining = len(self.stream_buffer.cur_chunk)
                if remaining > thr:
                    time.sleep(0.01)
                    continue
            try:
                t_obs_start = time.monotonic()
                # A.2 异步流水线 (§7.9): worker 在后台预备 obs, main 直接取.
                # worker 在上一 cycle 的 forward 期间已经准备好 obs (~15ms 任务),
                # 这里 get() 几乎 0 等. main 完成的总 cycle 不再含 obs_construct 时间.
                # Fallback: queue 空 (worker 启动初期 / sensor data 不全) → 同步取.
                # C.2: 通过 tuple 侧通道拿 head 帧 stamp, 准确测 image_age (obs dict 不动).
                obs_stamp_ns = None
                if self._pipelined_obs and self._obs_prefetch is not None:
                    try:
                        obs, obs_stamp_ns = self._obs_prefetch.get_obs(timeout=0.1)
                    except queue.Empty:
                        # worker 还没准备好 (启动初期 / sensor 不齐), 同步走一次
                        obs, obs_stamp_ns = self._get_observation_with_stamp()
                else:
                    obs, obs_stamp_ns = self._get_observation_with_stamp()
                if obs is None:
                    time.sleep(0.01)
                    continue
                t_obs_end = time.monotonic()

                # ── RTC guidance payload ──
                # Only inject when RTC is on AND we have a previous chunk. The
                # first call after load / flush / observe→execute has prev=None,
                # so sample_actions falls back to base_step (pi0_rtc.py:351).
                # inference_delay is estimated from last measured latency in
                # action-horizon steps (publish_rate=30Hz → 1 step ≈ 33ms).
                # Track whether RTC guidance is active THIS iteration (for log)
                rtc_active = False
                rtc_d_steps = 0
                rtc_exec_h = 0
                pc_for_log = None
                if self._enable_rtc:
                    with self._rtc_lock:
                        pc = self._rtc_prev_chunk.copy() if self._rtc_prev_chunk is not None else None
                    if pc is not None:
                        d_steps = int(max(0, round(self._last_infer_ms / 1000.0 * self.publish_rate)))
                        # R1 fix: normalize prev_chunk to match internal model space.
                        # pc is in raw un-normalized robot space (14-dim joint angles);
                        # sample_actions compares it against x_1 which is in normalized
                        # 32-dim model space. Apply (pc - mean) / std over first 14 dims
                        # so guidance error is computed like-for-like. Pi0RTC pads 14→32
                        # with zeros; dim_mask filters padding out of the guidance error
                        # (pi0_rtc.py:320-321), so sending 14-dim normalized is correct.
                        if self._norm_action_mean is not None:
                            pc_send = ((pc - self._norm_action_mean) / self._norm_action_std).astype(np.float32)
                        else:
                            pc_send = pc
                        obs['prev_action_chunk'] = pc_send
                        obs['inference_delay'] = d_steps
                        obs['execute_horizon'] = int(self._rtc_execute_horizon)
                        obs['max_guidance_weight'] = float(self._rtc_max_guidance_weight)
                        rtc_active = True
                        rtc_d_steps = d_steps
                        rtc_exec_h = int(self._rtc_execute_horizon)
                        pc_for_log = pc  # raw pc (for diag MAE comparison in raw space)
                        # mask_prefix_delay intentionally not forwarded — it's a
                        # Python bool that, once inside sample_actions' JIT boundary,
                        # becomes a tracer and breaks the `if mask_prefix_delay:`
                        # branch at pi0_rtc.py:323. Uses function default (False).

                if self._trace is not None:
                    self._trace_seq += 1
                    obs['trace_seq'] = self._trace_seq   # server 回填同 seq → 离线 join
                t_ws_send = time.monotonic()
                result = self.policy.infer(obs)
                t_ws_recv = time.monotonic()
                actions = result.get('actions', None)
                infer_ms = (t_ws_recv - t_start) * 1000
                self._last_infer_ms = infer_ms

                # EE-mode: server emits [H,16] world EE (action_kind="ee"). IK to
                # [H,14] joint (link6, R1) HERE — everything downstream (RTC snapshot,
                # integrate_new_chunk, smoothing, jump-protect, publish) stays
                # joint-domain → legacy joint path untouched. (xvla_inference_bringup §4 C)
                if (self._execution_mode == 'ee_pose' and isinstance(result, dict)
                        and result.get('action_kind') == 'ee'
                        and actions is not None and len(actions) > 0):
                    ee_chunk = np.asarray(actions, dtype=np.float32)
                    self._publish_ee_monitor(ee_chunk)
                    if self._ee_ctrl == 'firmware':
                        # ③ 不做主机 IK: 转 base-frame [xyz,rpy,grip]×2 (14D), 下发 PosCmd → 固件 IK。
                        # 仍走 stream_buffer/publish 复用 RTC 连续 cadence; 单位为 EE (m/rad)。
                        actions = self._ee_chunk_to_pose(ee_chunk)   # [H,14] EE-cmd
                    else:
                        actions = self._ee_chunk_to_joint(ee_chunk)  # [H,14] joint (host IK)
                    if self._trace is not None:
                        self._trace_record_infer(obs, result, ee_chunk, actions,
                                                 t_ws_send, t_ws_recv, infer_ms)

                if actions is not None and len(actions) > 0:
                    # Always snapshot the latest chunk for (a) RTC guidance on the
                    # next call and (b) the diagnostic MAE log below. Keeping it
                    # populated even when enable_rtc is off lets us measure the
                    # "natural chunk-to-chunk similarity" baseline as a control
                    # against the RTC-on measurement — otherwise we can't tell
                    # whether ratio < 1 is guidance or just horizon-dependent drift.
                    with self._rtc_lock:
                        self._rtc_prev_chunk = np.asarray(actions, dtype=float).copy()

                    # RACE GUARD: replay_mode may have flipped to 'replay' while
                    # policy.infer() was running (~500ms). _enter_replay_mode just
                    # filled cur_chunk with the entire episode (e.g. 681 frames);
                    # if we now call integrate_new_chunk with this 50-step policy
                    # chunk, StreamActionBuffer REPLACES cur_chunk entirely
                    # (line 215), wiping the replay buffer. Skip integration when
                    # mode has switched mid-flight; the policy chunk gets dropped.
                    if self._replay_mode == 'replay':
                        self.get_logger().info(
                            '[REPLAY] discarding mid-flight policy chunk '
                            '(replay_mode flipped to replay during inference)')
                        t_buffer_done = time.monotonic()
                    else:
                        self.stream_buffer.integrate_new_chunk(
                            actions,
                            max_k=self.latency_k,
                            min_m=self.min_smooth_steps)
                        t_buffer_done = time.monotonic()

                    # B1 latency profile sink (no-op when env var off).
                    if self._lat_profile_fp is not None:
                        self._record_latency_sample(
                            t_loop_start=t_start,
                            t_obs_start=t_obs_start, t_obs_end=t_obs_end,
                            t_ws_send=t_ws_send, t_ws_recv=t_ws_recv,
                            t_buffer_done=t_buffer_done,
                            obs=obs, result=result,
                            obs_stamp_ns=obs_stamp_ns,  # C.2: head 帧 stamp 走侧通道
                        )

                    # Publish full chunk for rerun_viz to show predicted trajectory
                    try:
                        from std_msgs.msg import Float32MultiArray, MultiArrayDimension
                        chunk_msg = Float32MultiArray()
                        chunk_msg.layout.dim = [
                            MultiArrayDimension(label='steps', size=actions.shape[0],
                                                stride=actions.shape[0] * actions.shape[1]),
                            MultiArrayDimension(label='dims', size=actions.shape[1],
                                                stride=actions.shape[1]),
                        ]
                        chunk_msg.data = actions.astype(np.float32).flatten().tolist()
                        self.pub_action_chunk.publish(chunk_msg)
                    except Exception as e:
                        self.get_logger().debug(f'action_chunk publish error: {e}')

                    # Print inference result summary to console
                    a0 = actions[0]
                    a_last = actions[-1]
                    self.get_logger().info(
                        f'infer {infer_ms:.0f}ms | chunk={actions.shape} | '
                        f'L[0]=[{a0[0]:+.2f},{a0[1]:+.2f},{a0[2]:+.2f},{a0[3]:+.2f},'
                        f'{a0[4]:+.2f},{a0[5]:+.2f},g={a0[6]:.2f}] '
                        f'R[0]=[{a0[7]:+.2f},{a0[8]:+.2f},{a0[9]:+.2f},{a0[10]:+.2f},'
                        f'{a0[11]:+.2f},{a0[12]:+.2f},g={a0[13]:.2f}] '
                        f'| Δ(0→{len(actions)-1}): L={np.linalg.norm(a_last[:6]-a0[:6]):.2f} '
                        f'R={np.linalg.norm(a_last[7:13]-a0[7:13]):.2f}')

                    # ── Chunk-alignment diagnostic ──
                    # Measure MAE(new, prev) separately in the RTC guidance window
                    # [d, exec_h) vs the free tail [exec_h, end). This is logged
                    # EVERY cycle, regardless of whether RTC guidance was actually
                    # injected this iteration (rtc_active flag). That way we can
                    # compare:
                    #   rtc=on  + injected=1 → measures guidance effect + natural drift
                    #   rtc=off + injected=0 → measures natural drift only (baseline)
                    # If ratio drops significantly when injected=1 vs injected=0,
                    # the guidance is actually pulling. If ratio is the same,
                    # "ratio<1" was just horizon-dependent chunk-decay.
                    if pc_for_log is None:
                        # Fall back to using the just-captured chunk as "prev" from
                        # one step ago — which is what we'll use next iteration too.
                        # On the very first iteration (no prev yet), skip.
                        pass
                    # Use the _rtc_prev_chunk BEFORE we overwrote it above. We need
                    # the PREVIOUS chunk for this iteration's diag. Easiest: cache
                    # it before the overwrite. Done via pc_for_log when rtc was
                    # active; for rtc-off case we have no pc_for_log. Workaround:
                    # also capture a local prev even when rtc_active was false.
                    try:
                        # Re-read: if we had pc_for_log (rtc injected), use it;
                        # otherwise use what we stored last time under a separate
                        # diagnostic attr that mirrors _rtc_prev_chunk every cycle.
                        diag_prev = pc_for_log if pc_for_log is not None else getattr(self, '_diag_last_chunk', None)
                        if diag_prev is not None:
                            n_steps = min(diag_prev.shape[0], actions.shape[0])
                            d_ = min(rtc_d_steps if rtc_active else 8, n_steps)  # for rtc-off use nominal d=8
                            eh = min(rtc_exec_h if rtc_active else 16, n_steps)
                            pc14 = diag_prev[:n_steps, :14]
                            new14 = actions[:n_steps, :14]
                            guid_mae = (
                                float(np.mean(np.abs(new14[d_:eh] - pc14[d_:eh])))
                                if eh > d_ else float('nan'))
                            free_mae = (
                                float(np.mean(np.abs(new14[eh:] - pc14[eh:])))
                                if n_steps > eh else float('nan'))
                            ratio = (guid_mae / free_mae) if (free_mae and free_mae > 1e-9) else float('nan')
                            pc_mean_abs = float(np.mean(np.abs(pc14)))
                            new_mean_abs = float(np.mean(np.abs(new14)))
                            if rtc_active:
                                inj = 'injected-norm' if self._norm_action_mean is not None else 'injected-raw'
                            else:
                                inj = 'baseline'
                            self.get_logger().info(
                                f'[{inj}] d={d_} exec_h={eh} | '
                                f'guid_MAE={guid_mae:.4f} free_MAE={free_mae:.4f} '
                                f'ratio={ratio:.2f} | |prev|={pc_mean_abs:.3f} |new|={new_mean_abs:.3f}')
                        # Stash current chunk for next iteration's baseline diag
                        self._diag_last_chunk = np.asarray(actions, dtype=float).copy()
                    except Exception as e:
                        self.get_logger().debug(f'chunk-diag error: {e}')

                    # Rerun: queue predicted trajectory for FK timer + log latency
                    # Pred queue is always safe (consumed by executor timer).
                    # Direct rr.log only if set_thread_local succeeded (_rr_infer_thread);
                    # otherwise route through the event queue so inference_ms still appears.
                    if self._use_rerun and self._rr is not None:
                        try:
                            ros_now_sec = self.get_clock().now().nanoseconds * 1e-9
                            # Queue FK-heavy predicted trajectory for the timer thread
                            if self._fk is not None:
                                self._rr_pred_queue.append((actions.copy(), ros_now_sec))
                            # Log inference latency
                            if self._rr_infer_thread:
                                self._rr.set_time("ros_time", timestamp=ros_now_sec)
                                self._rr.log("timeseries/inference_ms",
                                             self._rr.Scalars([float(infer_ms)]))
                            else:
                                # Fallback: route through event queue for executor thread
                                ms = float(infer_ms)
                                self._rr_event_queue.append((
                                    "timeseries/inference_ms",
                                    lambda v=ms: self._rr.Scalars([v]),
                                    ros_now_sec,
                                ))
                        except Exception as e:
                            self.get_logger().debug(f'Rerun prediction log error: {e}')

                # 记录整周期时长 (infer+IK+integrate), 供 ② 自适应 gate 预留步数。
                # 仅 infer 迭代到达此处 (gate-poll 迭代已 continue)。
                self._last_cycle_s = time.monotonic() - t_start

                # Re-read inference_rate each iteration so rtc_apply.sh / ros2
                # param set takes effect without restart.
                # ② 开环执行时不按 inference_rate 节流 —— 由顶部 buffer gate 决定何时重推。
                if not (self._open_loop_chunk and self._execution_enabled):
                    elapsed = time.monotonic() - t_start
                    period_live = 1.0 / max(0.1, float(self.inference_rate))
                    sleep_time = max(0, period_live - elapsed)
                    time.sleep(sleep_time)

            except Exception as e:
                import traceback
                self.get_logger().error(f'Inference error: {e}\n{traceback.format_exc()}')
                time.sleep(1.0)

    # ── Action publishing ───────────────────────────────────────────

    def _publish_action(self):
        """Timer callback: pop smoothed action and publish."""
        with self._exec_lock:
            enabled = self._execution_enabled
        if not enabled:
            self._last_published_action = None  # reset on observe so first execute re-seeds
            return
        act = self.stream_buffer.pop_next_action()
        if act is None:
            return

        # ③ firmware 模式: act 是 EE-cmd [xyz,rpy,grip]×2 (非关节)。直接发 PosCmd → 固件 IK,
        # 跳过下面关节域的 jump-protect/EMA (单位不同会误判); 安全护栏在 _publish_pos_cmd 内 (xyz 跳变)。
        if self._execution_mode == 'ee_pose' and self._ee_ctrl == 'firmware':
            self._publish_pos_cmd(act)
            self._last_published_action = act.copy()
            return

        # Jump protection: compare against current joint state (first action after
        # observe→execute) or last published action (subsequent actions)
        reference = self._last_published_action
        if reference is None:
            # First action after observe→execute: compare against current joints
            with self._sensor_lock:
                jl = self._joint_left_deque[-1] if self._joint_left_deque else None
                jr = self._joint_right_deque[-1] if self._joint_right_deque else None
            if jl is not None and jr is not None:
                ql = np.array(jl.position[:7])
                qr = np.array(jr.position[:7])
                if len(ql) >= 7 and len(qr) >= 7:
                    reference = np.concatenate([ql, qr])
        if reference is not None:
            max_jump = np.max(np.abs(act[:14] - reference[:14]))
            if max_jump > self._MAX_JOINT_JUMP_RAD:
                self.get_logger().warn(
                    f'Jump protection: rejected action with max Δ={np.degrees(max_jump):.1f}° '
                    f'(limit={np.degrees(self._MAX_JOINT_JUMP_RAD):.0f}°), flushing buffer')
                self._flush_stale_buffer()
                return
        # Layer 1.1E — EMA post-process on publish timeline. Orthogonal to 1.1B
        # (chunk-edge smoothing). When α<1, cmd[t] = α*cmd[t] + (1-α)*last_published,
        # attenuating high-freq cmd noise that exceeds Piper PD LP bandwidth.
        # IMPORTANT — gripper joints (j6, j13) are SKIPPED: gripper is effectively
        # binary (0=close, 0.07=open) and EMA-smoothing it traps cmd in mid-range
        # → physical "half-grasp release-then-regrab" failure (validated ep33).
        if self._publish_smooth_alpha < 1.0 and self._last_published_action is not None:
            alpha = self._publish_smooth_alpha
            prev = self._last_published_action[:14]
            raw_act = act[:14].copy()
            act[:14] = alpha * act[:14] + (1.0 - alpha) * prev
            # restore gripper to raw model output (no smoothing)
            act[6] = raw_act[6]
            act[13] = raw_act[13]
        self._last_published_action = act.copy()

        left = act[:7].copy()
        right = act[7:14].copy()
        # gripper_offset is an autonomy-only correction: policy outputs the value
        # the model was *trained* to predict, but the hardware closes a few mm
        # tighter than the trained target — subtract offset to compensate.
        # Replay path's actions were *recorded* from raw master-arm joint state
        # (no offset applied during teleop), so subtracting here would close the
        # gripper an extra `gripper_offset` mm tighter than the original
        # demonstration — destroys faithfulness, particularly bad for corner
        # grasping tasks where mm-scale precision matters.
        # Gripper close-snap (flag-gated, on RAW model output, BEFORE gripper_offset
        # so the threshold matches "模型输出 < thresh"). 软阈值: 仅把 < thresh 的部分
        # 裁成 0, ≥ thresh 原样通过 (开侧/释放轨迹保真). 修旧 action==state ckpt 抓软
        # 物到位后保持力≈0 脱手的问题: snap 到 0 → 固件保持位置误差 → 顶到电流上限 →
        # 夹紧力. replay 模式跳过 (与 gripper_offset 同理由, 保持示范忠实)。
        if self._grip_close_snap and self._replay_mode != 'replay':
            if left[6] < self._grip_close_snap_thr:
                left[6] = 0.0
            if right[6] < self._grip_close_snap_thr:
                right[6] = 0.0
        gripper_corr = 0.0 if self._replay_mode == 'replay' else self.gripper_offset
        left[6] = max(0.0, left[6] - gripper_corr)
        right[6] = max(0.0, right[6] - gripper_corr)

        now = self.get_clock().now().to_msg()

        # Publish to /master/joint_left and /master/joint_right
        for pub, values in [(self.pub_left, left), (self.pub_right, right)]:
            msg = JointState()
            msg.header.stamp = now
            msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            msg.position = values.tolist()
            pub.publish(msg)

        # Also publish combined action on /policy/actions
        msg = JointState()
        msg.header.stamp = now
        msg.name = [f'left_j{i}' for i in range(7)] + [f'right_j{i}' for i in range(7)]
        msg.position = act.tolist()
        self.pub_action.publish(msg)

    # ──────────────────────────────────────────────────────────────────────
    # EE-mode (multimodal protocol §B.5 / xvla_inference_bringup §4 C)
    # Active only when execution_mode=ee_pose. Server emits [H,16] world EE;
    # we IK to [H,14] joint (link6, R1) BEFORE the StreamActionBuffer, so all
    # smoothing/RTC/jump-protection/publish stay joint-domain — legacy joint
    # path is byte-identical when off.
    # ──────────────────────────────────────────────────────────────────────
    def _setup_ee_kinematics(self):
        import os as _os
        import sys as _sys
        repo = _os.path.dirname(_os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__)))))  # ros2_ws/src/piper/scripts → repo
        calib_dir = next((c for c in (_os.path.join(repo, 'calib'),
                                      '/data1/tim/workspace/deepdive_kai0/calib')
                          if _os.path.isfile(_os.path.join(c, 'piper_dh_ik.py'))), None)
        if calib_dir is None:
            self.get_logger().error('[ee] calib/piper_dh_ik.py not found → fallback to joint passthrough')
            self._execution_mode = 'joint'; return
        if calib_dir not in _sys.path:
            _sys.path.insert(0, calib_dir)
        try:
            # DH-model IK (CalFK link6) — 与训练 joint_to_ee6d 同模型; ikpy/URDF 有 ~5cm
            # 偏差不可用 (xvla_inference_bringup §4 C)。
            from piper_dh_ik import PiperDHIK
            self._ik = PiperDHIK()
            self._ee_seed_w = float(self.get_parameter('ee_ik_seed_weight').value)
            self._ee_soft_pos = float(self.get_parameter('ee_ik_soft_pos').value)
            self._ee_soft_rot = float(self.get_parameter('ee_ik_soft_rot').value)
            self._ee_max_jump = float(self.get_parameter('ee_ik_max_jump').value)
            self._ee_ctrl = str(self.get_parameter('ee_ctrl').value)
            import yaml as _yaml
            # NOTE: `repo` (from __file__ dirname-walk) lands on ros2_ws/install under
            # colcon symlink-install, so repo/config/calibration.yml does not exist.
            # Derive from calib_dir (robustly resolved above; its parent is the repo root).
            cyaml = (self.get_parameter('calibration_yaml').value
                     or _os.path.join(_os.path.dirname(calib_dir), 'config', 'calibration.yml'))
            tfs = _yaml.safe_load(open(cyaml)).get('transforms', {})
            # Dedicated attrs (NOT self._T_world_base{L,R}): the Rerun-viz init block in
            # __init__ runs AFTER this setup and re-defaults self._T_world_base{L,R}=None,
            # then only re-fills them via _load_calibration_and_fk when enable_rerun=True.
            # Under autonomy_launch rerun viz is a SEPARATE node so policy_inference's own
            # enable_rerun=False → those stay None → inv(None) crash. Decouple entirely.
            self._ee_T_world_baseL = np.array(tfs['T_world_baseL'], dtype=np.float64)
            self._ee_T_world_baseR = np.array(tfs['T_world_baseR'], dtype=np.float64)
        except Exception as e:
            self.get_logger().error(f'[ee] kinematics/calib init failed: {e} → fallback to joint')
            self._execution_mode = 'joint'; self._ik = None; return
        from geometry_msgs.msg import PoseStamped  # noqa: F401 (lazy, ee-only)
        from std_msgs.msg import Float32           # noqa: F401
        self.pub_ee_left = self.create_publisher(PoseStamped, '/policy/actions_ee_left', 10)
        self.pub_ee_right = self.create_publisher(PoseStamped, '/policy/actions_ee_right', 10)
        self.pub_grip_left = self.create_publisher(Float32, '/policy/actions_gripper_left', 10)
        self.pub_grip_right = self.create_publisher(Float32, '/policy/actions_gripper_right', 10)
        # ③ firmware 模式: PosCmd → /pos_cmd_{left,right} (launch remap 到各臂 arm_reader → EndPoseCtrl)
        self.pub_pos_left = self.create_publisher(PosCmd, '/pos_cmd_left', 10)
        self.pub_pos_right = self.create_publisher(PosCmd, '/pos_cmd_right', 10)
        self.get_logger().info(
            f'[ee] ee_pose ready: ee_ctrl={self._ee_ctrl} '
            f'({"firmware EndPoseCtrl (无主机IK)" if self._ee_ctrl=="firmware" else "host PiperDHIK→joint"}), calib={cyaml}')

    def _current_joints14(self):
        with self._sensor_lock:
            jl = self._joint_left_deque[-1] if self._joint_left_deque else None
            jr = self._joint_right_deque[-1] if self._joint_right_deque else None
        if jl is None or jr is None:
            return None
        ql, qr = np.array(jl.position[:7]), np.array(jr.position[:7])
        if len(ql) < 7 or len(qr) < 7:
            return None
        return np.concatenate([ql, qr]).astype(np.float64)

    def _ee8_to_base_link6(self, ee8, T_world_base):
        """8D world [xyz, quat_wxyz, grip] → T_base_link6 (4×4)."""
        from scipy.spatial.transform import Rotation
        xyz = np.asarray(ee8[:3], dtype=np.float64)
        q = np.asarray(ee8[3:7], dtype=np.float64)  # wxyz
        Rm = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        Tw = np.eye(4); Tw[:3, :3] = Rm; Tw[:3, 3] = xyz
        return np.linalg.inv(T_world_base) @ Tw

    def _ee_chunk_to_pose(self, ee_chunk):
        """③ [H,16] world EE → [H,14] base-frame EE-cmd [xyz(m),rpy(rad),grip(m)]×2。
        发 PosCmd → 固件 EndPoseCtrl (无主机 IK)。rpy 用 'xyz' 欧拉 (= CalFK/固件约定, 已 frame 验证)。"""
        from scipy.spatial.transform import Rotation
        H = ee_chunk.shape[0]
        out = np.zeros((H, 14), dtype=np.float32)
        for h in range(H):
            T6L = self._ee8_to_base_link6(ee_chunk[h, 0:8], self._ee_T_world_baseL)
            T6R = self._ee8_to_base_link6(ee_chunk[h, 8:16], self._ee_T_world_baseR)
            out[h, 0:3] = T6L[:3, 3]
            out[h, 3:6] = Rotation.from_matrix(T6L[:3, :3]).as_euler('xyz')
            out[h, 6] = ee_chunk[h, 7]      # L gripper (m)
            out[h, 7:10] = T6R[:3, 3]
            out[h, 10:13] = Rotation.from_matrix(T6R[:3, :3]).as_euler('xyz')
            out[h, 13] = ee_chunk[h, 15]    # R gripper (m)
        return out

    def _trace_record_infer(self, obs, result, ee_chunk, pose14, t_ws_send, t_ws_recv, infer_ms):
        """pipeline trace: 落 client 侧一条 — 发出的 obs / 收到的 16D chunk / 转出的 14D pose / 图。
        只在 self._trace 非空时由调用方触发; 这里再 try/except 兜底, 永不影响控制循环。"""
        try:
            seq = int(obs.get('trace_seq', self._trace_seq))
            st = np.asarray(obs.get('state'), dtype=np.float32).reshape(-1)
            ee = np.asarray(ee_chunk, dtype=np.float32)
            po = np.asarray(pose14, dtype=np.float32)
            srv_t = result.get('server_timing') if isinstance(result, dict) else None
            self._trace.event(
                stage='client_infer', seq=seq, t_mono=t_ws_recv,
                t_ws_send=t_ws_send, t_ws_recv=t_ws_recv, infer_ms=float(infer_ms),
                server_timing=srv_t,
                action_kind=result.get('action_kind') if isinstance(result, dict) else None,
                ee_ctrl=self._ee_ctrl, execute=bool(self._execution_enabled),
                chunk_h=int(ee.shape[0]), state14=st.tolist(),
                ee_recv_xyzL_h0=ee[0, 0:3].tolist(), ee_recv_xyzR_h0=ee[0, 8:11].tolist(),
                quat_norm_h0=[float(np.linalg.norm(ee[0, 3:7])), float(np.linalg.norm(ee[0, 11:15]))],
                pose14_h0=po[0].tolist())
            self._trace.save_arrays(seq, state14_sent=st, ee_chunk_recv=ee, pose14=po)
            imgs = obs.get('images') or {}
            for slot in ('top_head', 'hand_right', 'hand_left'):
                if slot in imgs:
                    self._trace.save_image(seq, slot, imgs[slot])
        except Exception as e:
            self.get_logger().warn(f'[trace] infer record failed: {e}')

    def _publish_pos_cmd(self, act):
        """③ 把一帧 EE-cmd [xyz,rpy,grip]×2 发成 PosCmd → /pos_cmd_{left,right} → 固件 EndPoseCtrl。
        简单安全护栏: 相对上次命令 xyz 跳变 > 8cm 则丢弃该帧 (防异常目标突跳)。"""
        if self.pub_pos_left is None:
            return
        # ③ re-binarize 夹爪: 撤掉 stream_buffer 线性平滑在二值夹爪上引入的中段值, snap 回
        # {0=闭, open_m=张} (阈值 open_m/2), 防 half-grasp。in-place 改 act → 调用方存的
        # _last_published_action 也是 snap 后的值, 一致。连续夹爪模型 ee_gripper_binarize=False 跳过。
        if self._ee_grip_binarize:
            thr = self._ee_grip_open_m * 0.5
            act[6] = self._ee_grip_open_m if act[6] > thr else 0.0
            act[13] = self._ee_grip_open_m if act[13] > thr else 0.0
        prev = self._last_published_action
        if prev is not None and len(prev) >= 14:
            dL = float(np.linalg.norm(np.asarray(act[0:3]) - np.asarray(prev[0:3])))
            dR = float(np.linalg.norm(np.asarray(act[7:10]) - np.asarray(prev[7:10])))
            if max(dL, dR) > 0.08:
                self.get_logger().warn(f'[ee-fw] PosCmd xyz 跳变 {max(dL,dR)*1000:.0f}mm > 80mm → 丢帧')
                if self._trace is not None:
                    self._trace.event(stage='poscmd', t_mono=time.monotonic(), dropped=True,
                                      jump_mm=float(max(dL, dR) * 1000.0),
                                      n=getattr(self, '_ee_fw_pub_n', 0))
                return
        for pub, a in ((self.pub_pos_left, act[0:7]), (self.pub_pos_right, act[7:14])):
            m = PosCmd()
            m.x, m.y, m.z = float(a[0]), float(a[1]), float(a[2])
            m.roll, m.pitch, m.yaw = float(a[3]), float(a[4]), float(a[5])
            m.gripper = float(a[6])
            pub.publish(m)
        if self._trace is not None:
            self._trace.event(stage='poscmd', t_mono=time.monotonic(), dropped=False,
                              n=getattr(self, '_ee_fw_pub_n', 0) + 1,
                              L=[float(x) for x in act[0:7]], R=[float(x) for x in act[7:14]])
        # 首发 + 每 ~3s 节流确认 (verify firmware path live; 单位 m/rad)
        self._ee_fw_pub_n = getattr(self, '_ee_fw_pub_n', 0) + 1
        if self._ee_fw_pub_n == 1 or self._ee_fw_pub_n % int(max(1, self.publish_rate * 3)) == 0:
            self.get_logger().info(
                f'[ee-fw] PosCmd #{self._ee_fw_pub_n} → /pos_cmd_{{l,r}}: '
                f'L xyz={np.round(act[0:3],3)} rpy={np.round(np.degrees(act[3:6]),0)} g={act[6]:.3f} | '
                f'R xyz={np.round(act[7:10],3)} g={act[13]:.3f}')

    def _ee_chunk_to_joint(self, ee_chunk):
        """[H,16] world EE → [H,14] joint via per-step seeded DH IK (link6). None if no seed."""
        seed = self._ee_last_good if self._ee_last_good is not None else self._current_joints14()
        if seed is None:
            self.get_logger().warn('[ee] no joint seed yet (waiting /puppet/joint_*) — drop chunk')
            return None
        H = ee_chunk.shape[0]
        out = np.zeros((H, 14), dtype=np.float32)
        qL, qR = seed[:6].copy(), seed[7:13].copy()
        w = getattr(self, '_ee_seed_w', 0.0)
        soft_pos = getattr(self, '_ee_soft_pos', 0.04)
        soft_rot = getattr(self, '_ee_soft_rot', 0.12)
        max_jump = getattr(self, '_ee_max_jump', 0.15)

        # 解一步: 纯位姿 IK(可达目标精确)+ 关节空间限速(rate-limit)。返回 (q, used, far)
        #   每步把关节变化 clamp 到 ±max_jump → 既不翻飞(有界, 止姿态乱)也不冻结
        #   (始终朝目标 slew, 止断续)。seed 用上一步限速后的 q → 无级联冻结。
        #   far = 位姿残差超软阈(模型目标真够不到), 仅作诊断计数, 不改变限速行为。
        def _step(T6, qseed):
            s, _hard = self._ik.solve(T6, qseed, seed_weight=w)
            pr, rr_ = self._ik.last_pos_res, self._ik.last_rot_res
            far = (pr > soft_pos or rr_ > soft_rot)
            delta = np.clip(s - qseed, -max_jump, max_jump)
            return qseed + delta, (not far), far

        failL = failR = 0                  # 硬容差 (5mm) 内才算 0; 这里统计真 hold 数
        pres_max = rres_max = 0.0
        jumpL = jumpR = 0.0
        prevL = qL.copy(); prevR = qR.copy()
        for h in range(H):
            T6L = self._ee8_to_base_link6(ee_chunk[h, 0:8], self._ee_T_world_baseL)
            qL, usedL, heldL = _step(T6L, qL)
            if heldL:
                failL += 1
            pres_max = max(pres_max, self._ik.last_pos_res); rres_max = max(rres_max, self._ik.last_rot_res)
            jumpL = max(jumpL, float(np.abs(qL - prevL).max())); prevL = qL.copy()

            T6R = self._ee8_to_base_link6(ee_chunk[h, 8:16], self._ee_T_world_baseR)
            qR, usedR, heldR = _step(T6R, qR)
            if heldR:
                failR += 1
            pres_max = max(pres_max, self._ik.last_pos_res); rres_max = max(rres_max, self._ik.last_rot_res)
            jumpR = max(jumpR, float(np.abs(qR - prevR).max())); prevR = qR.copy()

            out[h, :6] = qL; out[h, 6] = ee_chunk[h, 7]
            out[h, 7:13] = qR; out[h, 13] = ee_chunk[h, 15]
        self._ee_last_good = out[-1].astype(np.float64)
        spanL = float(np.ptp(ee_chunk[:, 0:3], axis=0).max())
        spanR = float(np.ptp(ee_chunk[:, 8:11], axis=0).max())
        bndL = float(np.abs(out[0, :6] - seed[:6]).max())
        bndR = float(np.abs(out[0, 7:13] - seed[7:13]).max())
        self.get_logger().info(
            f'[ee-diag] far(unreach) L{failL}/{H} R{failR}/{H} | res pos<={pres_max*1000:.0f}mm rot<={np.degrees(rres_max):.1f}deg '
            f'| eeSpan L{spanL*1000:.0f} R{spanR*1000:.0f}mm | jump/step L{np.degrees(jumpL):.0f} R{np.degrees(jumpR):.0f}deg '
            f'| bnd L{np.degrees(bndL):.0f} R{np.degrees(bndR):.0f}deg | softPos={soft_pos*1000:.0f}mm rateLim={np.degrees(max_jump):.0f}deg/step')
        return out

    def _publish_ee_monitor(self, ee_chunk):
        """Publish ee_chunk[0] world pose + gripper for monitoring (control still via joint)."""
        if self.pub_ee_left is None:
            return
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Float32
        now = self.get_clock().now().to_msg()
        e = ee_chunk[0]
        for pub, gpub, base, gi in ((self.pub_ee_left, self.pub_grip_left, 0, 7),
                                    (self.pub_ee_right, self.pub_grip_right, 8, 15)):
            ps = PoseStamped()
            ps.header.stamp = now; ps.header.frame_id = 'world'
            ps.pose.position.x = float(e[base + 0]); ps.pose.position.y = float(e[base + 1])
            ps.pose.position.z = float(e[base + 2])
            ps.pose.orientation.w = float(e[base + 3]); ps.pose.orientation.x = float(e[base + 4])
            ps.pose.orientation.y = float(e[base + 5]); ps.pose.orientation.z = float(e[base + 6])
            pub.publish(ps); gpub.publish(Float32(data=float(e[gi])))


def main():
    rclpy.init()
    node = PolicyInferenceNode()
    # Rerun set_time+log is not atomic — all ROS2 callbacks and timers MUST run
    # on the same thread. Use SingleThreadedExecutor unconditionally.
    executor = rclpy.executors.SingleThreadedExecutor()
    if node._rr_require_single_thread and isinstance(
            executor, rclpy.executors.MultiThreadedExecutor):
        node.get_logger().fatal(
            'Rerun is enabled but executor is MultiThreadedExecutor. '
            'Rerun set_time+log is non-atomic and WILL corrupt timelines. '
            'Use SingleThreadedExecutor or disable Rerun.')
        raise RuntimeError('Rerun requires SingleThreadedExecutor')
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
