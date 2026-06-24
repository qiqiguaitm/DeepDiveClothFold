#!/usr/bin/env python3
"""谨慎重标定【单个关节】零点 — 只动一条臂的一个关节, 其余不碰。

⚠️ 持久化固件写入 (JointConfig 0xAE)。会影响真机坐标系 → 部署敏感。
   只在确诊某关节漂移、且该关节已【物理对到机械零位刻线】时运行。
   - 改 master 关节: 不动 slave → 部署坐标系不变 (最安全)。
   - 改 slave 关节: 等于改部署坐标系 → 必须对到与训练一致的机械刻线,
     之后用 diag_left_arm_align.py / 训练参考帧复核 slave home 位姿读数回到:
       J1 -9.3°, J2 +11.7°, J3 -29.1°, J4 +4.8°, J5 +43.2°, J6 -10.2°

流程 (照 SDK piper_set_joint_zero 官方): 失能该电机 → 人手对到刻线 → 回车 → JointConfig(N,0xAE) → 使能。
用法: python3 rezero_one_joint.py --can can_left_mas --joint 3
      (--joint 1..6; 默认 --dry 仅打印不写, 加 --commit 才真正写固件)
"""
import argparse
import time

from piper_sdk import C_PiperInterface_V2


def read_deg(p):
    js = p.GetArmJointMsgs().joint_state
    return [getattr(js, f"joint_{i}") / 1000.0 for i in range(1, 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can", required=True, help="can_left_mas / can_left_slave / ...")
    ap.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    ap.add_argument("--commit", action="store_true", help="真正写固件 (默认 dry-run 只读+提示)")
    args = ap.parse_args()

    p = C_PiperInterface_V2(args.can); p.ConnectPort(); time.sleep(0.2)
    print(f"[{args.can}] 当前关节(度): " + " ".join(f"J{i+1}={v:.2f}" for i, v in enumerate(read_deg(p))))
    print(f"目标: 仅重标定 J{args.joint} 零点。")
    if not args.commit:
        print("\n[dry-run] 未写固件。确认无误后加 --commit 重跑。")
        return
    if input(f"\n确认对 {args.can} 的 J{args.joint} 写零点? 先确保该关节已对到机械刻线 [yes/no]: ") != "yes":
        print("已取消, 未写入。"); return
    p.DisableArm(args.joint)
    print(f"J{args.joint} 已失能。请把该关节【物理对到机械零位刻线】并扶稳。")
    if input("对好后回车设零 (输 q 放弃): ") == "q":
        p.EnableArm(args.joint); print("已放弃, 重新使能。"); return
    p.JointConfig(args.joint, 0xAE)
    time.sleep(0.2)
    p.EnableArm(args.joint)
    time.sleep(0.3)
    print(f"J{args.joint} 零点已写入。复核当前: " + " ".join(f"J{i+1}={v:.2f}" for i, v in enumerate(read_deg(p))))
    print("→ 再跑 diag_left_arm_align.py 确认主从在刻线处该关节都 ≈ 0。")


if __name__ == "__main__":
    main()
