#!/usr/bin/env python3
"""READ-ONLY 左臂主从对齐诊断 — 不使能、不发运动、不写固件。

用途: 定位左臂哪个关节(肘 J3 / 腕 J4·J5)的主从零点漂移, 以及漂移在 master 还是 slave。

用法:
  1) 先停掉 teleop / 所有 piper ROS 节点 (让两臂自由, 但 CAN 保持 UP)。
  2) 把左主臂、左从臂的【每个关节都对到机械零位刻线/凹槽】(拖动到位; 失能拖动可手动)。
  3) 运行本脚本, 读两臂关节反馈 (单位 度)。在机械刻线处, 正确标定的臂每个关节应 ≈ 0。
     哪条臂的某关节读数明显非 0, 那条臂的那个关节就是漂移的 → 只重标定它。
  本脚本只调用 ConnectPort + GetArmJointMsgs (纯读), 绝不 Enable/JointCtrl/JointConfig。
"""
import time

from piper_sdk import C_PiperInterface_V2

MAS = "can_left_mas"
SLA = "can_left_slave"
JNAMES = ["J1", "J2", "J3(肘)", "J4(腕)", "J5(腕)", "J6"]


def read_deg(p):
    js = p.GetArmJointMsgs().joint_state
    raw = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
    # Piper 关节反馈单位 = 0.001°
    return [r / 1000.0 for r in raw]


def limit_flags(p):
    """各关节 angle-limit 标志 (1=该关节到角度限位, 跟不动主臂的常见原因)。"""
    try:
        e = p.GetArmStatus().arm_status.err_status
        return [int(getattr(e, f"joint_{i}_angle_limit")) for i in range(1, 7)]
    except Exception:
        return [-1] * 6


def main():
    m = C_PiperInterface_V2(MAS); m.ConnectPort()
    s = C_PiperInterface_V2(SLA); s.ConnectPort()
    time.sleep(0.2)
    print(f"{'':8}" + "".join(f"{n:>10}" for n in JNAMES))
    print("-" * 70)
    try:
        while True:
            md = read_deg(m); sd = read_deg(s)
            dd = [a - b for a, b in zip(md, sd)]
            print(f"{'master':8}" + "".join(f"{v:>10.2f}" for v in md))
            print(f"{'slave ':8}" + "".join(f"{v:>10.2f}" for v in sd))
            print(f"{'Δ(m-s)':8}" + "".join(f"{v:>10.2f}" for v in dd))
            # 标记最大偏差关节
            i = max(range(6), key=lambda k: abs(dd[k]))
            ml = limit_flags(m); sl = limit_flags(s)
            print(f"limit旗 m:{ml}  s:{sl}   (1=该关节到限位→跟不动)")
            print(f"  最大主从偏差: {JNAMES[i]} = {dd[i]:+.2f}°"
                  f"   (Δ 在各位姿恒定→零点漂移可重标; Δ 随位姿变/限位=1→机械或限位问题)")
            print("-" * 70)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n退出 (未做任何写入/运动)")


if __name__ == "__main__":
    main()
