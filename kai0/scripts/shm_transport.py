"""POSIX shared-memory transport for V1 client↔server (替 msgpack + TCP loopback).

设计 (§7.8 中 C.4 / "POSIX SHM ring buffer"):
- 两个 POSIX shm 区域 (client→server request, server→client response).
- 每个区域 = [fixed 64B header] + [image region (zero-copy memcpy)] + [metadata msgpack].
- 同步: busy-poll on 64-bit seq counter (50µs latency, 比 semaphore wake-up 快).
- 图像走 zero-copy memcpy: 3 张 224x224x3 RGB uint8 = 451KB, 跳过 msgpack 编解码 (-2-3ms).
- 元数据 (state / prompt / prev_action_chunk / 小 dict) 仍走 msgpack (~5KB, 0.1ms).

兼容性:
- 仅 V1 路径用 (start_autonomy_v1.sh transport:=shm). 默认 WS, JAX 不受影响.
- Server / Client 任一边失败 → 上层 fallback 到 WS.

P95 减幅估计:
- TCP loopback transport: ~2ms
- Image msgpack encode + decode: ~3-4ms
- Total: 5-7ms cycle P95 ↓
"""

from __future__ import annotations

import os
import time
import struct
import threading
from multiprocessing import shared_memory
from typing import Optional

import msgpack
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Header layout (固定 64 字节, 跨进程小心 endianness)
# ────────────────────────────────────────────────────────────────────────────
# struct.format 完整 = '<QIIIIQ32x' (little-endian, 8+4+4+4+4+8+32 padding = 64 bytes)
#   seq_no:       uint64  — atomic-ish counter, 客户端 +1 后写, 服务端 poll
#   payload_size: uint32  — image + metadata 总长 (sanity check)
#   image_size:   uint32  — image bytes (3 * 224*224*3 = 451200 expected)
#   metadata_size:uint32  — msgpack-encoded metadata 长度
#   flags:        uint32  — bit0=READY, bit1=ERROR (reserved)
#   timestamp_ns: uint64  — client 写时 monotonic, debug 用
HEADER_FMT = '<QIIIIQ32x'
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 64

FLAG_READY = 1
FLAG_ERROR = 2

# Default region sizes — 4 MB for request, 64 KB for response
DEFAULT_REQ_SIZE = 4 * 1024 * 1024
DEFAULT_RESP_SIZE = 64 * 1024

# Image region 固定: 3 cameras × CHW 224x224x3 uint8 = 451200 bytes
IMAGE_REGION_SIZE = 3 * 3 * 224 * 224
assert IMAGE_REGION_SIZE == 451584  # CHW: (3, 224, 224) per camera × 3 cams

# Default shm names (POSIX shm_open under the hood). 在 /dev/shm/ 下能 ls 看到.
DEFAULT_REQ_NAME = 'kai0_v1_obs'
DEFAULT_RESP_NAME = 'kai0_v1_chunk'


def _pack_header(seq, payload_size, image_size, metadata_size, flags, ts_ns):
    return struct.pack(HEADER_FMT, seq, payload_size, image_size, metadata_size, flags, ts_ns)


def _unpack_header(buf):
    return struct.unpack(HEADER_FMT, bytes(buf[:HEADER_SIZE]))


def _try_unlink(name):
    """Best-effort unlink stale shm region by name (POSIX, /dev/shm/<name>)."""
    try:
        existing = shared_memory.SharedMemory(name=name)
        existing.close()
        existing.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        # 已经 unlinked 或别的, 忽略
        print(f"[shm_transport] _try_unlink({name}) note: {e}")


# ────────────────────────────────────────────────────────────────────────────
# ShmServer — serve_policy_v1.py 用
# ────────────────────────────────────────────────────────────────────────────
class ShmServer:
    """Server side: create both shm regions, poll for new requests, write responses.

    Use case:
        server = ShmServer(infer_callback=lambda obs: policy.infer(obs))
        server.start()    # spawns daemon poll thread
        ...
        server.stop()
    """

    def __init__(
        self,
        infer_callback,                # callable(obs_dict) -> result_dict
        req_name: str = DEFAULT_REQ_NAME,
        resp_name: str = DEFAULT_RESP_NAME,
        req_size: int = DEFAULT_REQ_SIZE,
        resp_size: int = DEFAULT_RESP_SIZE,
        logger=None,
    ):
        self._infer = infer_callback
        self._req_name = req_name
        self._resp_name = resp_name
        self._log = logger or _PrintLogger()

        # Clean stale + create fresh
        _try_unlink(req_name)
        _try_unlink(resp_name)
        self._req_shm = shared_memory.SharedMemory(name=req_name, create=True, size=req_size)
        self._resp_shm = shared_memory.SharedMemory(name=resp_name, create=True, size=resp_size)
        # Zero out headers
        self._req_shm.buf[:HEADER_SIZE] = b'\x00' * HEADER_SIZE
        self._resp_shm.buf[:HEADER_SIZE] = b'\x00' * HEADER_SIZE
        self._log.info(
            f'[ShmServer] regions created: req=/dev/shm/{req_name} ({req_size}B), '
            f'resp=/dev/shm/{resp_name} ({resp_size}B)')

        self._running = False
        self._thread = None
        self._last_seen_seq = 0
        self._n_served = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name='ShmServerPoll')
        self._thread.start()
        self._log.info('[ShmServer] poll loop started')

    def stop(self):
        self._running = False
        # daemon thread dies with process
        try:
            self._req_shm.close()
            self._req_shm.unlink()
            self._resp_shm.close()
            self._resp_shm.unlink()
            self._log.info(f'[ShmServer] stopped, served={self._n_served}')
        except Exception as e:
            self._log.warn(f'[ShmServer] cleanup error: {e}')

    def _loop(self):
        # SHM v2: hybrid poll — 200µs hard spin (覆盖 client-side jitter), 然后 backoff sleep.
        # Hard spin 期间不 release GIL → 最低 detect latency. Backoff 期间释放 CPU.
        idle_start_ns = time.monotonic_ns()
        while self._running:
            try:
                seq, payload_size, image_size, metadata_size, flags, ts_ns = _unpack_header(
                    self._req_shm.buf)
                if seq > self._last_seen_seq and (flags & FLAG_READY):
                    self._handle_request(seq, image_size, metadata_size)
                    self._last_seen_seq = seq
                    self._n_served += 1
                    idle_start_ns = time.monotonic_ns()  # 刚处理完, idle 计时重置
                else:
                    idle_ns = time.monotonic_ns() - idle_start_ns
                    if idle_ns < 200_000:           # 0-200µs: 硬 spin, 不 yield GIL
                        continue
                    elif idle_ns < 30_000_000:      # 0.2-30ms: yield GIL but don't sleep
                        time.sleep(0)
                    else:                           # > 30ms: 没新请求很久, soft sleep
                        time.sleep(0.0005)          # 500µs
            except Exception as e:
                self._log.warn(f'[ShmServer] loop error: {e}')
                time.sleep(0.01)

    def _handle_request(self, seq: int, image_size: int, metadata_size: int):
        # Image: zero-copy view of req_shm buffer at fixed offset.
        # CHW uint8, 3 cameras concatenated, layout = [front, right, left] × (3, 224, 224).
        img_off = HEADER_SIZE
        meta_off = img_off + IMAGE_REGION_SIZE
        if image_size != IMAGE_REGION_SIZE:
            raise ValueError(
                f'image_size={image_size} != expected {IMAGE_REGION_SIZE}')
        image_buf = self._req_shm.buf[img_off:img_off + image_size]
        # Reshape as np ndarray view (3 cams, 3 channels, 224, 224).
        # 注意: np.frombuffer 返 read-only view, .copy() 不需要 — 服务端不修改
        all_imgs = np.frombuffer(image_buf, dtype=np.uint8).reshape(3, 3, 224, 224)

        # Metadata msgpack
        meta_bytes = bytes(self._req_shm.buf[meta_off:meta_off + metadata_size])
        metadata = msgpack.unpackb(meta_bytes, raw=False)

        # Assemble obs dict (匹配 WS path 的 schema)
        obs = {
            'images': {
                'top_head':   all_imgs[0],  # numpy view (3, 224, 224) uint8
                'hand_right': all_imgs[1],
                'hand_left':  all_imgs[2],
            },
            'state':  np.asarray(metadata['state'], dtype=np.float32),
            'prompt': metadata['prompt'],
        }
        # Optional fields (RTC + extended modalities)
        for k in ('prev_action_chunk', 'inference_delay', 'execute_horizon',
                  'rtc_max_guidance_weight', 'dataset_id'):
            if k in metadata:
                obs[k] = metadata[k]
                if k == 'prev_action_chunk' and obs[k] is not None:
                    obs[k] = np.asarray(obs[k], dtype=np.float32)
        # Depth / EE pose (extended modalities, V1 path 默认不开)
        if 'depth_top_head' in metadata:
            obs['depth_top_head'] = np.asarray(metadata['depth_top_head'], dtype=np.float32)
        if 'ee_pose_left' in metadata:
            obs['ee_pose_left'] = np.asarray(metadata['ee_pose_left'], dtype=np.float32)
            obs['ee_pose_right'] = np.asarray(metadata['ee_pose_right'], dtype=np.float32)

        # Infer
        result = self._infer(obs)

        # Pack response
        result_bytes = msgpack.packb(self._sanitize_result(result), use_bin_type=True)
        rsize = len(result_bytes)
        if HEADER_SIZE + rsize > self._resp_shm.size:
            raise ValueError(f'response {rsize}B too big for shm {self._resp_shm.size}B')
        self._resp_shm.buf[HEADER_SIZE:HEADER_SIZE + rsize] = result_bytes
        # Write header LAST (seq is the "publish" signal for client)
        hdr = _pack_header(seq, rsize, 0, rsize, FLAG_READY, time.monotonic_ns())
        self._resp_shm.buf[:HEADER_SIZE] = hdr

    @staticmethod
    def _sanitize_result(result):
        """Convert numpy → list/bytes for msgpack."""
        if not isinstance(result, dict):
            return result
        out = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                out[k] = {'__nd__': True, 'dtype': str(v.dtype),
                          'shape': list(v.shape), 'data': v.tobytes()}
            elif isinstance(v, dict):
                out[k] = ShmServer._sanitize_result(v)
            else:
                out[k] = v
        return out


# ────────────────────────────────────────────────────────────────────────────
# ShmClient — policy_inference_node.py 用 (替 WebsocketClientPolicy.infer)
# ────────────────────────────────────────────────────────────────────────────
class ShmClient:
    """Client side: attach to existing shm regions, write request + poll response.

    Mimics WebsocketClientPolicy.infer(obs) API.
    """

    def __init__(
        self,
        req_name: str = DEFAULT_REQ_NAME,
        resp_name: str = DEFAULT_RESP_NAME,
        attach_timeout_sec: float = 30.0,
        logger=None,
    ):
        self._log = logger or _PrintLogger()
        # Wait for server to create regions (poll with timeout)
        deadline = time.monotonic() + attach_timeout_sec
        while True:
            try:
                self._req_shm = shared_memory.SharedMemory(name=req_name)
                self._resp_shm = shared_memory.SharedMemory(name=resp_name)
                break
            except FileNotFoundError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f'shm regions {req_name}/{resp_name} not present after '
                        f'{attach_timeout_sec}s — is server up?')
                time.sleep(0.1)
        self._log.info(
            f'[ShmClient] attached: req=/dev/shm/{req_name} ({self._req_shm.size}B), '
            f'resp=/dev/shm/{resp_name} ({self._resp_shm.size}B)')
        self._seq = 0
        self._lock = threading.Lock()  # 防止多 thread 抢 infer

    def infer(self, obs: dict, timeout_sec: float = 5.0) -> dict:
        """Send obs to server via shm, busy-poll for response, return result dict."""
        with self._lock:
            self._seq += 1
            seq = self._seq

            # SHM v2 optimization: zero-copy image write via numpy view on shm buffer.
            # 旧版: np.stack (alloc + 3 memcpy) + tobytes (1 memcpy) + slice assign
            #       (1 memcpy) = 5 memcpy + 1 alloc.
            # 新版: numpy view onto shm + np assignment = 3 memcpy 直写 shm, 无中间
            #       alloc. 省 0.5-1ms.
            img_off = HEADER_SIZE
            # View shm buffer as ndarray (3 cams, 3 channels, 224, 224) uint8
            img_view = np.frombuffer(
                self._req_shm.buf[img_off:img_off + IMAGE_REGION_SIZE],
                dtype=np.uint8
            ).reshape(3, 3, 224, 224)
            images = obs['images']
            # numpy assignment = single memcpy per camera into shm
            img_view[0] = np.asarray(images['top_head'], dtype=np.uint8)
            img_view[1] = np.asarray(images['hand_right'], dtype=np.uint8)
            img_view[2] = np.asarray(images['hand_left'], dtype=np.uint8)

            # Pack metadata (everything except images)
            metadata = {
                'state':  np.asarray(obs['state']).tolist(),
                'prompt': obs.get('prompt', ''),
            }
            for k in ('prev_action_chunk', 'inference_delay', 'execute_horizon',
                      'rtc_max_guidance_weight', 'dataset_id'):
                if k in obs:
                    v = obs[k]
                    if isinstance(v, np.ndarray):
                        metadata[k] = v.tolist()
                    else:
                        metadata[k] = v
            for k in ('depth_top_head', 'ee_pose_left', 'ee_pose_right'):
                if k in obs:
                    v = obs[k]
                    if isinstance(v, np.ndarray):
                        metadata[k] = v.tolist()
                    else:
                        metadata[k] = v

            meta_bytes = msgpack.packb(metadata, use_bin_type=True)
            msize = len(meta_bytes)
            meta_off = img_off + IMAGE_REGION_SIZE
            if meta_off + msize > self._req_shm.size:
                raise ValueError(
                    f'metadata {msize}B too big for shm (req_size={self._req_shm.size}, '
                    f'meta_off={meta_off})')
            self._req_shm.buf[meta_off:meta_off + msize] = meta_bytes

            # Write header LAST (seq is the publish signal)
            hdr = _pack_header(
                seq, IMAGE_REGION_SIZE + msize, IMAGE_REGION_SIZE, msize,
                FLAG_READY, time.monotonic_ns())
            self._req_shm.buf[:HEADER_SIZE] = hdr

            # SHM v2: hybrid poll for response — busy-spin during expected forward window,
            # backoff to yield/sleep after that. forward 通常 35ms, 用 0-30ms spin+yield.
            deadline = time.monotonic() + timeout_sec
            poll_start_ns = time.monotonic_ns()
            while True:
                rseq, rpayload, rimg, rmeta, rflags, rts = _unpack_header(self._resp_shm.buf)
                if rseq == seq:
                    if rflags & FLAG_ERROR:
                        raise RuntimeError('server returned error flag')
                    resp_bytes = bytes(self._resp_shm.buf[HEADER_SIZE:HEADER_SIZE + rmeta])
                    raw = msgpack.unpackb(resp_bytes, raw=False)
                    return self._reconstruct(raw)
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f'shm response seq={seq} not seen after {timeout_sec}s (last seen seq={rseq})')
                elapsed_ns = time.monotonic_ns() - poll_start_ns
                if elapsed_ns < 200_000:        # 0-200µs: 硬 spin
                    continue
                elif elapsed_ns < 50_000_000:   # 0.2-50ms: yield GIL (其他线程可跑) 不真 sleep
                    time.sleep(0)
                else:                            # > 50ms: 真 sleep (forward 异常慢)
                    time.sleep(0.0001)

    @staticmethod
    def _reconstruct(data):
        """Inverse of ShmServer._sanitize_result (rebuild numpy arrays)."""
        if isinstance(data, dict):
            if data.get('__nd__'):
                return np.frombuffer(data['data'], dtype=np.dtype(data['dtype'])).reshape(
                    data['shape']).copy()
            return {k: ShmClient._reconstruct(v) for k, v in data.items()}
        if isinstance(data, list):
            return [ShmClient._reconstruct(v) for v in data]
        return data


class _PrintLogger:
    """Tiny fallback logger when ROS2/Python logger not passed in."""
    def info(self, m): print(f"[shm_transport] INFO  {m}")
    def warn(self, m): print(f"[shm_transport] WARN  {m}")
    def error(self, m): print(f"[shm_transport] ERROR {m}")
