#!/usr/bin/env python3
"""X-VLA 真机 trace: 模型输出(指令 EE) vs 机械臂实际(achieved EE) 跟踪分析。

从 trace 的 rosbag 读:
  指令  = /pos_cmd_{left,right}            (piper_msgs/PosCmd, node 下发, 模型输出转成的 base EE)
  实际  = /puppet/end_pose_euler_{left,right} (piper_msgs/PosCmd, 固件回报当前 EE, 同表示/同系)
两者同为 [x,y,z(m), roll,pitch,yaw(rad), gripper] → 直接可比。

输出 (每臂):
  • 指令运动幅度 (各轴 range + 路径长) — 判模型是否真在动 / 还是近静止
  • 跟踪误差 (achieved 插值到指令时刻): xyz mm / rpy deg 的 p50/p95/max
  • 跟踪滞后 (主运动轴上 cmd vs achieved 互相关峰值 → 估计固件跟踪延迟)
  • 指令/实际 速率 (Hz)

用法: python xvla/analyze_tracking.py <trace_dir>   (需 source ROS + 工作区 install)
"""
from __future__ import annotations
import sys, os
import numpy as np

CYAN, GREEN, YELLOW, RED, NC = "\033[0;36m", "\033[0;32m", "\033[1;33m", "\033[0;31m", "\033[0m"


WANT = ["/pos_cmd_left", "/pos_cmd_right",
        "/puppet/end_pose_euler_left", "/puppet/end_pose_euler_right"]


def _read_bag(bag_dir):
    """读 rosbag mcap → {topic: (N,8) [t_s, x,y,z, r,p,yaw, grip]}。
    用 mcap_ros2 (按 mcap 内嵌 schema 解码, 不需装 piper_msgs)。
    若缺库: pip install mcap mcap-ros2-support (或 PYTHONPATH 指向已装目录)。"""
    import glob
    try:
        from mcap.reader import NonSeekingReader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:
        print(f"{RED}缺 mcap_ros2 → pip install mcap mcap-ros2-support{NC}", file=sys.stderr)
        raise
    wantset = set(WANT)
    out = {t: [] for t in WANT}
    factory = DecoderFactory()
    # NonSeekingReader 顺序读, 不碰 summary/footer; log_time_order=False 不预排序 → 增量产出。
    # 未干净收尾的 mcap 末尾索引记录会截断 → 捕获 EndOfFile, 保留已读到的全部消息。
    for mc in sorted(glob.glob(os.path.join(bag_dir, "*.mcap"))):
        cache = {}
        with open(mc, "rb") as f:
            try:
                for schema, channel, message in NonSeekingReader(f).iter_messages(log_time_order=False):
                    if channel.topic not in wantset:
                        continue
                    dec = cache.get(channel.id)
                    if dec is None:
                        dec = factory.decoder_for(channel.message_encoding, schema)
                        cache[channel.id] = dec
                    m = dec(message.data)
                    out[channel.topic].append((message.log_time * 1e-9,
                                               m.x, m.y, m.z, m.roll, m.pitch, m.yaw, m.gripper))
            except Exception as e:
                # 未干净收尾的 mcap 末尾索引/footer 记录会截断 (EndOfFile / struct.error /
                # RecordLengthLimitExceeded 等) — 数据消息在前面已读完, 安全保留已读部分。
                print(f"{YELLOW}  (mcap 尾部截断, 忽略: {type(e).__name__}){NC}", file=sys.stderr)
    return {k: np.asarray(v, dtype=float) for k, v in out.items() if v}


def _analyze_arm(name, cmd, ach):
    print(f"\n{CYAN}── {name} ──{NC}")
    if cmd is None or ach is None or len(cmd) < 3 or len(ach) < 3:
        print(f"  {YELLOW}数据不足 (cmd={0 if cmd is None else len(cmd)}, ach={0 if ach is None else len(ach)}){NC}")
        return
    t0 = min(cmd[0, 0], ach[0, 0])
    tc, ta = cmd[:, 0] - t0, ach[:, 0] - t0
    pc, pa = cmd[:, 1:8], ach[:, 1:8]   # x y z r p y g

    # 速率
    hz_c = (len(tc) - 1) / max(1e-6, tc[-1] - tc[0])
    hz_a = (len(ta) - 1) / max(1e-6, ta[-1] - ta[0])
    print(f"  速率: 指令 {hz_c:.1f}Hz ({len(tc)} 帧)  实际 {hz_a:.1f}Hz ({len(ta)} 帧)  时长 {tc[-1]-tc[0]:.1f}s")

    # 指令运动幅度 (模型在动吗)
    rng = pc[:, 0:3].max(0) - pc[:, 0:3].min(0)
    path = float(np.sum(np.linalg.norm(np.diff(pc[:, 0:3], axis=0), axis=1)))
    print(f"  指令 xyz 幅度: X={rng[0]*1000:.0f} Y={rng[1]*1000:.0f} Z={rng[2]*1000:.0f} mm  "
          f"路径长={path*1000:.0f}mm  ({'近静止' if path < 0.02 else '有运动'})")
    rng_a = pa[:, 0:3].max(0) - pa[:, 0:3].min(0)
    print(f"  实际 xyz 幅度: X={rng_a[0]*1000:.0f} Y={rng_a[1]*1000:.0f} Z={rng_a[2]*1000:.0f} mm")
    g_c = pc[:, 6]
    print(f"  夹爪指令: min={g_c.min():.4f} max={g_c.max():.4f}  开/合切换次数="
          f"{int(np.sum(np.abs(np.diff((g_c > (g_c.min()+g_c.max())/2).astype(int)))))}")

    # 跟踪误差: achieved 插值到指令时刻
    ach_i = np.column_stack([np.interp(tc, ta, pa[:, k]) for k in range(7)])
    exyz = np.linalg.norm(pc[:, 0:3] - ach_i[:, 0:3], axis=1) * 1000.0  # mm
    def q(a): return f"p50={np.median(a):.1f} p95={np.percentile(a,95):.1f} max={a.max():.1f}"
    print(f"  跟踪误差 xyz(mm): {q(exyz)}")
    # 姿态误差: 用测地角 (geodesic), 而非 euler 分量差 — 后者在 gimbal/wrap 处虚高。
    # euler 'xyz' 与 node 生成时 as_euler('xyz') 同约定, from_euler 精确还原。
    try:
        from scipy.spatial.transform import Rotation
        Rc = Rotation.from_euler('xyz', pc[:, 3:6])
        Ra = Rotation.from_euler('xyz', ach_i[:, 3:6])
        erot_deg = (Rc * Ra.inv()).magnitude() * 180.0 / np.pi
        print(f"  跟踪误差 rot(deg, 测地): {q(erot_deg)}")
    except Exception:
        erpy = np.degrees(np.abs(((pc[:, 3:6] - ach_i[:, 3:6] + np.pi) % (2*np.pi) - np.pi)))
        print(f"  跟踪误差 rpy/轴(deg): R={q(erpy[:,0])} P={q(erpy[:,1])} Y={q(erpy[:,2])}")

    # 滞后: 主运动轴 (指令 range 最大者) 上 cmd vs achieved 互相关
    ax = int(np.argmax(rng))
    if rng[ax] > 0.01:
        # 在均匀网格上重采样 (用指令中位 dt)
        dt = np.median(np.diff(tc))
        grid = np.arange(tc[0], tc[-1], dt)
        c = np.interp(grid, tc, pc[:, ax]); a = np.interp(grid, ta, pa[:, ax])
        c -= c.mean(); a -= a.mean()
        if c.std() > 1e-6 and a.std() > 1e-6:
            xc = np.correlate(a, c, mode="full")
            lag = (np.argmax(xc) - (len(c) - 1)) * dt
            print(f"  跟踪滞后 (轴{'XYZ'[ax]}, 互相关): {lag*1000:+.0f}ms  (>0 = 实际滞后于指令)")

    # 跟踪健康判定
    ok = np.median(exyz) < 30 and np.percentile(exyz, 95) < 80
    print(f"  [{GREEN+'好'+NC if ok else YELLOW+'偏大'+NC}] 跟踪 (p50<30mm 且 p95<80mm 视为好)")


def main():
    if len(sys.argv) < 2:
        print("用法: python xvla/analyze_tracking.py <trace_dir>", file=sys.stderr); sys.exit(1)
    import glob
    d = sys.argv[1]
    bag = os.path.join(d, "rosbag")
    if not glob.glob(os.path.join(bag, "*.mcap")):
        print(f"{RED}rosbag 下无 *.mcap: {bag}{NC}", file=sys.stderr)
        sys.exit(2)
    print(f"{CYAN}══════════ 模型输出 vs 机械臂跟踪 ══════════{NC}\ntrace: {d}")
    data = _read_bag(bag)
    _analyze_arm("LEFT 左臂", data.get("/pos_cmd_left"), data.get("/puppet/end_pose_euler_left"))
    _analyze_arm("RIGHT 右臂", data.get("/pos_cmd_right"), data.get("/puppet/end_pose_euler_right"))
    print()


if __name__ == "__main__":
    main()
