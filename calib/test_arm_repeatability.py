#!/usr/bin/env python3
"""机械臂定位重复性测试 — 诊断 hand-eye 标定误差是否来自臂的物理重复性。

原理: 标定每帧的 T_base_ee 来自 FK(关节角)。若臂每次定位到"同一姿态"时
关节角/末端在抖, FK 就抖, 标定就散。本测试反复定位同一目标姿态, 量化末端散布,
并区分:
  - 异向逼近 (varied approach): 每次从不同姿态逼近目标 → 暴露回差(backlash)
  - 同向逼近 (same approach):   每次从同一姿态逼近目标 → 排除回差, 看纯噪声
若 异向 >> 同向 → 机械回差; 两者都大 → 编码器/控制噪声。
不连相机, 只动机械臂 + FK。

用法 (系统 python3, 需 piper_sdk + CAN up):
  python3 calib/test_arm_repeatability.py --arm left  --can can_left_slave
  python3 calib/test_arm_repeatability.py --arm right --can can_right_slave   # 基线对照

输出: 终端表格 + verify_out/repeatability_<arm>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('PIPER_SDK_DIR', '/home/tim/workspace/piper_sdk')
from piper_sdk import C_PiperInterface
from piper_fk import PiperFK

JOINT_FACTOR = 57295.7795        # rad → 0.001° (Piper SDK 内部单位)
SETTLE_THRESHOLD_DEG = 0.5       # 到位判断阈值
SETTLE_VEL_DEG_S = 2.0           # 静止速度阈值
SETTLE_WAIT_S = 0.5              # 到位后额外等待
SETTLE_TIMEOUT_S = 20.0          # 到位超时


class PiperArm:
    """Piper SDK 直连 (控制 + 读关节 + FK)。复刻自 capture_handeye.PiperArm,
    去掉相机依赖以便在无 pyrealsense 的系统 python 下运行。"""

    def __init__(self, can_name: str):
        self.piper = C_PiperInterface(can_name)
        self.piper.ConnectPort()
        self.fk = PiperFK()

    def read_joints_rad(self) -> np.ndarray:
        """读取当前 6 关节角 (rad)。"""
        js = self.piper.GetArmJointMsgs().joint_state
        mdeg = np.array([js.joint_1, js.joint_2, js.joint_3,
                         js.joint_4, js.joint_5, js.joint_6], dtype=np.float64)
        return mdeg / JOINT_FACTOR

    def read_fk(self) -> np.ndarray:
        """当前末端 FK 4×4 (m/rad)。"""
        return self.fk.fk_homogeneous(self.read_joints_rad())

    def move_to(self, q_rad: np.ndarray, speed_pct: int = 30) -> None:
        """发送关节角指令 (rad)。EnablePiper 失败抛异常。"""
        enabled = False
        for _ in range(100):
            if self.piper.EnablePiper():
                enabled = True
                break
            time.sleep(0.01)
        if not enabled:
            raise RuntimeError("EnablePiper() failed — check CAN bus and power")
        self.piper.MotionCtrl_2(0x01, 0x01, speed_pct, 0x00)  # CAN ctrl, MOVE J
        time.sleep(0.05)
        ctrl = [int(q * JOINT_FACTOR) for q in q_rad]
        self.piper.JointCtrl(*ctrl)

    def wait_settled(self, target_rad: np.ndarray,
                     timeout_s: float = SETTLE_TIMEOUT_S) -> bool:
        """等待到位且静止 (err<阈值 且 连续3次速度<阈值), 然后额外等待。"""
        t0 = time.monotonic()
        prev = None
        stable = 0
        while time.monotonic() - t0 < timeout_s:
            cur = self.read_joints_rad()
            err = np.max(np.abs(np.degrees(cur - target_rad)))
            vel_ok = True
            if prev is not None:
                vel_ok = np.max(np.abs(np.degrees(cur - prev))) / 0.05 < SETTLE_VEL_DEG_S
            prev = cur.copy()
            if err < SETTLE_THRESHOLD_DEG and vel_ok:
                stable += 1
                if stable >= 3:
                    time.sleep(SETTLE_WAIT_S)
                    return True
            else:
                stable = 0
            time.sleep(0.05)
        return False


def load_target_poses(session_arm_dir: str, n_targets: int) -> list[np.ndarray]:
    """从 recalib 的 pose_*.npz 读 joint_angles, 均匀挑 n_targets 个代表姿态。"""
    files = sorted(f for f in os.listdir(session_arm_dir)
                   if f.startswith('pose_') and f.endswith('.npz'))
    qs = [np.load(os.path.join(session_arm_dir, f), allow_pickle=True)['joint_angles']
          for f in files]
    idx = np.linspace(0, len(qs) - 1, n_targets).round().astype(int)
    return [np.asarray(qs[i], dtype=np.float64) for i in idx]


def measure(arm: PiperArm, target: np.ndarray, approaches: list[np.ndarray],
            n: int, speed: int) -> dict:
    """重复定位 target n 次, 每次先移到 approaches[k] 再回 target, 记录散布。

    Returns dict: pos_rms_mm, ori_rms_deg, joint_std_deg [6], n。
    """
    xyz, rotvec, qrec = [], [], []
    for k in range(n):
        appr = approaches[k % len(approaches)]
        arm.move_to(appr, speed)
        arm.wait_settled(appr)
        arm.move_to(target, speed)
        if not arm.wait_settled(target):
            print(f"    [{k+1}/{n}] WARN: 未在 {SETTLE_TIMEOUT_S}s 内到位/静止, 仍记录", flush=True)
        T = arm.read_fk()
        xyz.append(T[:3, 3])
        rotvec.append(Rotation.from_matrix(T[:3, :3]).as_rotvec())
        qrec.append(arm.read_joints_rad())
        print(f"    [{k+1}/{n}] xyz=({T[0,3]*1000:.1f}, {T[1,3]*1000:.1f}, {T[2,3]*1000:.1f}) mm", flush=True)
    xyz = np.array(xyz)
    rotvec = np.array(rotvec)
    qrec = np.array(qrec)
    pos_rms = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()) * 1000.0)
    ori_rms = float(np.degrees(np.sqrt((np.linalg.norm(rotvec - rotvec.mean(0), axis=1) ** 2).mean())))
    return {'pos_rms_mm': pos_rms, 'ori_rms_deg': ori_rms,
            'joint_std_deg': np.degrees(qrec.std(0)).tolist(), 'n': n}


def main() -> None:
    ap = argparse.ArgumentParser(description="机械臂定位重复性测试")
    ap.add_argument('--arm', required=True, choices=['left', 'right'])
    ap.add_argument('--can', required=True, help='CAN 接口名 (如 can_left_slave)')
    ap.add_argument('--session', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data', 'recalib'))
    ap.add_argument('--n', type=int, default=20, help='每姿态重复次数')
    ap.add_argument('--n-targets', type=int, default=3, help='目标姿态数')
    ap.add_argument('--speed', type=int, default=30, help='运动速度百分比 (建议 20-30)')
    args = ap.parse_args()

    targets = load_target_poses(os.path.join(args.session, args.arm), args.n_targets)

    print("=" * 64)
    print(f"⚠️  机械臂重复性测试 — arm={args.arm} can={args.can}")
    print(f"   将让 {args.arm} 臂自动来回运动 {args.n_targets}姿态 × {args.n}次 × 2组(异向/同向)")
    print(f"   速度 {args.speed}%。请确认: 臂周围无人无障碍、急停可达、上电正常。")
    print("=" * 64)
    if input("确认安全并开始? 输入 yes 继续: ").strip().lower() != 'yes':
        print("已取消。")
        return

    arm = PiperArm(args.can)
    result = {'arm': args.arm, 'targets': []}
    for ti, target in enumerate(targets):
        print(f"\n=== 目标姿态 {ti+1}/{len(targets)} ===")
        # 异向逼近: 用其它目标姿态轮流作出发点
        varied = [targets[j] for j in range(len(targets)) if j != ti] or [target]
        # 同向逼近: 固定用同一个出发点 (回差对照)
        same = [varied[0]]
        print("  [异向逼近 varied approach]")
        a = measure(arm, target, varied, args.n, args.speed)
        print("  [同向逼近 same approach]")
        b = measure(arm, target, same, args.n, args.speed)
        result['targets'].append({'varied': a, 'same': b})
        print(f"  → 异向: 位置RMS={a['pos_rms_mm']:.2f}mm 朝向RMS={a['ori_rms_deg']:.2f}°")
        print(f"  → 同向: 位置RMS={b['pos_rms_mm']:.2f}mm 朝向RMS={b['ori_rms_deg']:.2f}°")
        worst = int(np.argmax(a['joint_std_deg']))
        print("  → 异向各关节散布(°): " +
              ' '.join(f"j{i+1}={s:.3f}" for i, s in enumerate(a['joint_std_deg'])) +
              f"  | 最大: j{worst+1}")

    # 输出到 session 目录 (与采集数据同处, 跑脚本的用户通常对其有写权限)
    out = os.path.join(args.session, f'repeatability_{args.arm}.json')
    with open(out, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f"汇总 ({args.arm} 臂)")
    pv = float(np.mean([t['varied']['pos_rms_mm'] for t in result['targets']]))
    ps = float(np.mean([t['same']['pos_rms_mm'] for t in result['targets']]))
    print(f"  平均位置RMS: 异向={pv:.2f}mm  同向={ps:.2f}mm")
    print("  判读: >2-3mm=重复性差(对上标定误差); 异向>>同向=机械回差; 两者都大=编码器/控制噪声")
    js = np.mean([t['varied']['joint_std_deg'] for t in result['targets']], axis=0)
    print("  平均各关节散布(°): " + ' '.join(f"j{i+1}={s:.3f}" for i, s in enumerate(js)) +
          f"  | 元凶关节: j{int(np.argmax(js))+1}")
    print(f"  已存: {out}")


if __name__ == '__main__':
    main()
