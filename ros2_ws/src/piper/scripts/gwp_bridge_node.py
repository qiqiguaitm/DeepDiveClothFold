#!/usr/bin/env python3
"""gwp_ans / gwp_ori → Piper ROS2 桥接节点。

把 GigaWorld-Policy 世界-动作模型 (gwp_ans, Wan2.2-5B 扩散) 接到现有 kai0 autonomy 栈:
复用 multi_camera_node / arm_reader_node 已发布的相机+关节话题,组装 gwp 观测,经 ZeroMQ
请求 scripts/serve_gwp_opt.py (gwp venv 内, fp8+T_a3, ~87ms),拿回 [48,14] 绝对关节动作,
经分块缓冲 + min-jerk 衔接 + EMA 平滑 + 跳变保护 + /policy/execute 闸,发到 /master/joint_*。

与 policy_inference_node 的关系:这是一个**独立、自包含**的发布节点 (不 import kai0 node),
用同一套话题契约,可由 launch 与相机/机械臂节点并起。gwp 模型推理在另一个 venv 的 server 进程里,
本节点只做 ROS I/O + 客户端 + 平滑 (轻量, 与 ROS2/torch-cpu 环境兼容)。

观测契约 (发给 server):
  observation.state                 [14]  左臂(6关节+夹爪) + 右臂(6关节+夹爪)
  observation.images.cam_high       CHW float[0,1]  ← /camera_f color (top_head)
  observation.images.cam_left_wrist CHW float[0,1]  ← /camera_l color (hand_left)
  observation.images.cam_right_wrist CHW float[0,1] ← /camera_r color (hand_right)
返回: {"actions": [48,14] 绝对关节目标, "server_timing": {...}}

依赖: rclpy, sensor_msgs, std_msgs, numpy, pyzmq, torch(cpu, 仅用于线格式 torch.save/load).
"""
import threading
from io import BytesIO

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32MultiArray

import zmq
import torch


# ---------- ZeroMQ 客户端 (与 giga_models.sockets 线格式一致: torch.save/load 一个 dict) ----------
class GwpClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 20000):
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.ctx = zmq.Context()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    @staticmethod
    def _to_bytes(obj):
        buf = BytesIO()
        torch.save(obj, buf)
        return buf.getvalue()

    @staticmethod
    def _from_bytes(data):
        return torch.load(BytesIO(data), map_location="cpu", weights_only=False)

    def infer(self, observation: dict):
        self.sock.send(self._to_bytes({"endpoint": "inference", "data": observation}))
        reply = self.sock.recv()
        if reply == b"ERROR":
            raise RuntimeError("gwp server returned ERROR")
        return self._from_bytes(reply)

    def ping(self) -> bool:
        try:
            self.sock.send(self._to_bytes({"endpoint": "ping"}))
            self.sock.recv()
            return True
        except zmq.error.ZMQError:
            # REQ socket stuck after timeout → reset
            self.sock.close(0)
            self._connect()
            return False


def _minjerk(n: int) -> np.ndarray:
    """quintic smoothstep 6t^5-15t^4+10t^3, 端点零速零加速 (与 kai0 node Layer1.1B 一致)。"""
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return 6 * t**5 - 15 * t**4 + 10 * t**3


class GwpBridgeNode(Node):
    def __init__(self):
        super().__init__("gwp_bridge_node")
        g = ReentrantCallbackGroup()

        # ---- params ----
        self.declare_parameter("server_host", "127.0.0.1")
        self.declare_parameter("server_port", 8093)
        self.declare_parameter("img_front_topic", "/camera_f/camera/color/image_raw")
        self.declare_parameter("img_left_topic", "/camera_l/camera/color/image_raw")
        self.declare_parameter("img_right_topic", "/camera_r/camera/color/image_raw")
        self.declare_parameter("puppet_left_topic", "/puppet/joint_left")
        self.declare_parameter("puppet_right_topic", "/puppet/joint_right")
        self.declare_parameter("action_chunk", 48)
        self.declare_parameter("exec_horizon", 16)          # 每块执行步数 (到点再 re-infer)
        self.declare_parameter("blend_steps", 6)            # 新旧块 min-jerk 衔接步
        self.declare_parameter("publish_rate", 30.0)        # Hz
        self.declare_parameter("publish_smooth_alpha", 0.7) # EMA: cmd=α·new+(1-α)·last
        self.declare_parameter("max_joint_jump_rad", 0.5)   # 单关节跳变上限 (~28.6°)
        self.declare_parameter("gripper_open_m", 0.08)
        self.declare_parameter("execute_default", False)

        gp = lambda k: self.get_parameter(k).value
        self.action_chunk = int(gp("action_chunk"))
        self.exec_horizon = int(gp("exec_horizon"))
        self.blend_steps = int(gp("blend_steps"))
        self.pub_rate = float(gp("publish_rate"))
        self.alpha = float(gp("publish_smooth_alpha"))
        self.max_jump = float(gp("max_joint_jump_rad"))
        self._execute = bool(gp("execute_default"))

        # ---- state ----
        self._lock = threading.Lock()
        self._img = {"cam_high": None, "cam_left_wrist": None, "cam_right_wrist": None}
        self._jl = None  # left joint [7]
        self._jr = None  # right joint [7]
        self._chunk = None        # current [48,14] absolute joints
        self._chunk_idx = 0
        self._prev_tail = None     # last published 14-vec, for min-jerk blend + EMA
        self._last_pub = None
        self._infer_busy = False

        self.client = GwpClient(gp("server_host"), int(gp("server_port")))

        # ---- subs ----
        self.create_subscription(Image, gp("img_front_topic"),
                                 lambda m: self._cb_img(m, "cam_high"), 1, callback_group=g)
        self.create_subscription(Image, gp("img_left_topic"),
                                 lambda m: self._cb_img(m, "cam_left_wrist"), 1, callback_group=g)
        self.create_subscription(Image, gp("img_right_topic"),
                                 lambda m: self._cb_img(m, "cam_right_wrist"), 1, callback_group=g)
        self.create_subscription(JointState, gp("puppet_left_topic"),
                                 self._cb_jl, 50, callback_group=g)
        self.create_subscription(JointState, gp("puppet_right_topic"),
                                 self._cb_jr, 50, callback_group=g)
        self.create_subscription(Bool, "/policy/execute", self._cb_execute, 1, callback_group=g)

        # ---- pubs ----
        self.pub_action = self.create_publisher(JointState, "/policy/actions", 10)
        self.pub_left = self.create_publisher(JointState, "/master/joint_left", 10)
        self.pub_right = self.create_publisher(JointState, "/master/joint_right", 10)
        self.pub_chunk = self.create_publisher(Float32MultiArray, "/policy/action_chunk", 5)

        # ---- loops ----
        self.create_timer(1.0 / self.pub_rate, self._publish_tick, callback_group=g)
        self.create_timer(0.01, self._infer_tick, callback_group=g)  # 试触发推理 (busy/idx 控制)
        self.get_logger().info(
            f"gwp_bridge up: server {gp('server_host')}:{gp('server_port')}, "
            f"chunk={self.action_chunk} exec_h={self.exec_horizon} execute={self._execute}")

    # ---------------- callbacks ----------------
    def _cb_img(self, msg: Image, key: str):
        # multi_camera_node 发布 rgb8; np.frombuffer 零拷贝 view → CHW float[0,1]
        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)[:, :, :3]
        chw = np.ascontiguousarray(arr.transpose(2, 0, 1)).astype(np.float32) / 255.0
        with self._lock:
            self._img[key] = torch.from_numpy(chw)

    def _cb_jl(self, msg: JointState):
        with self._lock:
            self._jl = np.asarray(msg.position[:7], dtype=np.float32)

    def _cb_jr(self, msg: JointState):
        with self._lock:
            self._jr = np.asarray(msg.position[:7], dtype=np.float32)

    def _cb_execute(self, msg: Bool):
        self._execute = bool(msg.data)
        self.get_logger().info(f"/policy/execute -> {self._execute}")

    # ---------------- observation ----------------
    def _build_obs(self):
        with self._lock:
            if self._jl is None or self._jr is None:
                return None
            if any(self._img[k] is None for k in self._img):
                return None
            state = np.concatenate([self._jl, self._jr]).astype(np.float32)  # [14]
            imgs = {f"observation.images.{k}": self._img[k].clone() for k in self._img}
        obs = {"observation.state": torch.from_numpy(state)}
        obs.update(imgs)
        return obs

    # ---------------- inference (re-plan) ----------------
    def _infer_tick(self):
        if self._infer_busy:
            return
        # 仅在没有计划或当前块已执行到 exec_horizon 时 re-infer
        with self._lock:
            need = self._chunk is None or self._chunk_idx >= self.exec_horizon
        if not need:
            return
        obs = self._build_obs()
        if obs is None:
            return
        self._infer_busy = True
        try:
            res = self.client.infer(obs)
            act = res["actions"]
            act = act.numpy() if hasattr(act, "numpy") else np.asarray(act)
            act = act.astype(np.float32)[: self.action_chunk, :14]
            with self._lock:
                self._chunk = act
                self._chunk_idx = 0
        except Exception as e:
            self.get_logger().warn(f"infer failed: {e}")
        finally:
            self._infer_busy = False

    # ---------------- publish ----------------
    def _publish_tick(self):
        with self._lock:
            if self._chunk is None:
                return
            idx = min(self._chunk_idx, self.action_chunk - 1)
            target = self._chunk[idx].copy()
            self._chunk_idx += 1

        # min-jerk 衔接: 新块前 blend_steps 步从 prev_tail 渐变到模型值
        if self._prev_tail is not None and self._chunk_idx <= self.blend_steps:
            w = _minjerk(self.blend_steps + 1)[self._chunk_idx]
            target = (1.0 - w) * self._prev_tail + w * target

        # EMA 发布平滑
        if self._last_pub is not None:
            target = self.alpha * target + (1.0 - self.alpha) * self._last_pub

        # 跳变保护: 任一关节相对上次发布跳变 > max_jump → clamp (而非拒绝, 保证连续)
        if self._last_pub is not None:
            d = target - self._last_pub
            big = np.abs(d) > self.max_jump
            if big.any():
                target[big] = self._last_pub[big] + np.sign(d[big]) * self.max_jump
                self.get_logger().warn(f"jump clamp on joints {np.where(big)[0].tolist()}")

        self._last_pub = target.copy()
        if self._chunk_idx >= self.exec_horizon:
            self._prev_tail = target.copy()

        # 始终发布 /policy/actions (可视化); 仅 execute 时发到 /master/joint_*
        self._pub_js(self.pub_action, target)
        if self._execute:
            self._pub_js(self.pub_left, target[:7])
            self._pub_js(self.pub_right, target[7:14])

    def _pub_js(self, pub, vec):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.position = [float(x) for x in vec]
        pub.publish(m)


def main():
    rclpy.init()
    node = GwpBridgeNode()
    from rclpy.executors import MultiThreadedExecutor
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
