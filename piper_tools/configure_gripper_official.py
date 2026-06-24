#!/usr/bin/env python3
# -*-coding:utf8-*-
"""官方夹爪标定: 把 4 个夹爪配成规范 0-70mm frame (max_range=70 + 机械全闭处设零点).

背景
----
本机 4 个夹爪出厂/历史配置是 max_range_config=100 (非官方), 且零点不在机械全闭处
(从臂命令 0 闭不到底)。官方规范是 max_range=70mm、command 0 = 机械全闭。配成规范后:
主臂读 0(闭)..70mm(开), 从臂命令 0(闭)..70mm(开), 主→从 **1:1 直通即可**, 不再需要
任何软件 remap。

⚠️ 这会改变夹爪坐标系 → 旧数据集/旧 ckpt 的夹爪维度需按新 70mm range 重算 norm_stats
后才能部署。属持久化固件写入(set_zero 可再次重设, 无出厂回退)。

官方 API
--------
  GripperTeachingPendantParamConfig(100, 70, 1) + ArmParamEnquiryAndConfig(4)  # 设 max_range=70
  GripperCtrl(0, 1000, 0x00, 0xAE)                                            # 当前位置设为 0 点

设零点为何 4 只都要手动
----
set_zero(0xAE) 只在夹爪【失能(0x00)】时生效。从臂一失能就被内部弹力顶开几 mm
(右臂实测回弹 3mm), 自动驱动顶死后失能必回弹 → 零点偏开 → 命令 0 闭不到底;
带力(0x01)时发 0xAE 又被固件忽略。所以唯一可靠办法是【人手把夹爪压在机械硬底】
再设零(主、从同理)。运行时夹爪始终使能, 电机主动驱到零点(=硬底, 可克服回弹),
命令 0 必闭到底。

用法 (先停遥操: ./start_scripts/start_data_collect.sh stop)
----
  # 一次标全 4 只 (每只: 手把夹爪捏到硬底按住→回车→设零):
  python3 piper_tools/configure_gripper_official.py --role both --arm both
  # 只标从臂两只 (主臂之前标好了就不用重标):
  python3 piper_tools/configure_gripper_official.py --role slave --arm both
  # 只读核对当前 4 个夹爪参数:
  python3 piper_tools/configure_gripper_official.py --check
"""
import argparse
import sys
import time

from piper_sdk import C_PiperInterface_V2

ARMS = {
    "left": {"master": "can_left_mas", "slave": "can_left_slave"},
    "right": {"master": "can_right_mas", "slave": "can_right_slave"},
}
EFFORT = 1000
TARGET_RANGE = 70  # 官方 max_range_config (mm)


def angle_um(p):
    try:
        return int(p.GetArmGripperMsgs().gripper_state.grippers_angle)
    except Exception:
        return None


def read_max_range(p):
    try:
        p.ArmParamEnquiryAndConfig(0x04)
        time.sleep(0.2)
        tp = p.GetGripperTeachingPendantParamFeedback()
        return tp.arm_gripper_teaching_param_feedback.max_range_config
    except Exception:
        return None


def set_range_70(p, label):
    print(f"  [{label}] 设 max_range={TARGET_RANGE} ...")
    for _ in range(3):
        p.GripperTeachingPendantParamConfig(100, TARGET_RANGE, 1)
        time.sleep(0.2)
        p.ArmParamEnquiryAndConfig(0x04)
        time.sleep(0.3)
    mr = read_max_range(p)
    ok = (mr == TARGET_RANGE)
    print(f"  [{label}] max_range_config now = {mr}  {'OK' if ok else '!! 未生效'}")
    return ok


def set_zero_here(p, label):
    """主臂用: 失能后在当前(人手捏住机械底)位置设零点.

    主臂由人手压住机械底, 失能也不会回弹, 故按官方 demo 失能再设零。
    """
    print(f"  [{label}] 在当前位置设零点(失能) ...")
    p.GripperCtrl(0, EFFORT, 0x00, 0)      # 失能
    time.sleep(1.2)
    p.GripperCtrl(0, EFFORT, 0x00, 0xAE)   # 设零点
    time.sleep(1.0)
    a = angle_um(p)
    print(f"  [{label}] 设零后读数 = {a} ({(a or 0) / 1000:.2f} mm) (应≈0)")
    return a


ROLE_CN = {"master": "主", "slave": "从"}


def configure_one(side, role):
    """官方标定一只夹爪: 设 range=70, 然后**人手捏到机械硬底**时设零点.

    为什么 4 只都要手动: set_zero(0xAE) 只在失能(0x00)时生效; 而从臂一失能就回弹
    几 mm(右臂实测 3mm) → 零点偏开 → 命令 0 闭不到底。自动驱动顶死后失能必回弹,
    带力时发 0xAE 又被固件忽略。唯一可靠办法是人手把夹爪压在硬底再设零(主从同理)。
    运行时夹爪始终使能, 电机会主动驱到零点(即硬底, 能克服回弹弹力), 故命令 0 必闭到底。
    """
    can = ARMS[side][role]
    label = f"{side}-{role}"
    print(f"\n=== {label} ({can}) ===")
    p = C_PiperInterface_V2(can)
    p.ConnectPort()
    t0 = time.time()
    while angle_um(p) is None and time.time() - t0 < 5:
        time.sleep(0.1)
    if angle_um(p) is None:
        print(f"  [FAIL] {can} 无夹爪数据, 跳过")
        return False

    # 1) 设官方 range=70
    set_range_70(p, label)

    # 2) 失能夹爪 → 可手动开合
    p.GripperCtrl(0, EFFORT, 0x00, 0)
    time.sleep(0.3)

    # 3) 人手捏到机械硬底并按住 → 在该位置设零
    input(f"  >>> 用手把【{side} {ROLE_CN[role]}夹爪】捏合到机械【硬底】并【按住别松】, 然后按 Enter 设零...")
    set_zero_here(p, label)
    return True


def check_all():
    print("=== 当前 4 个夹爪参数 ===")
    for side in ("left", "right"):
        for role in ("master", "slave"):
            can = ARMS[side][role]
            try:
                p = C_PiperInterface_V2(can)
                p.ConnectPort()
                time.sleep(0.3)
                print(f"  {side}-{role:6s} {can:14s}: max_range={read_max_range(p)}  "
                      f"angle={angle_um(p)}")
            except Exception as e:  # noqa: BLE001
                print(f"  {side}-{role} {can}: ERROR {e}")


def main():
    ap = argparse.ArgumentParser(description="官方夹爪标定 (range=70 + set_zero)")
    ap.add_argument("--role", choices=["master", "slave", "both"], default="both")
    ap.add_argument("--arm", choices=["left", "right", "both"], default="both")
    ap.add_argument("--check", action="store_true", help="只读核对当前参数, 不写")
    args = ap.parse_args()

    if args.check:
        check_all()
        return

    sides = ["left", "right"] if args.arm == "both" else [args.arm]
    roles = ["master", "slave"] if args.role == "both" else [args.role]
    print("=" * 60)
    print(f"  官方夹爪标定: range={TARGET_RANGE} + 机械全闭设零点")
    print(f"  目标: arms={sides} roles={roles}")
    print("  前提: 已停遥操 (./start_scripts/start_data_collect.sh stop)")
    print("=" * 60)

    done = []
    for side in sides:
        for role in roles:
            if configure_one(side, role):
                done.append(f"{side}-{role}")

    print(f"\n[完成] 已标定: {done}")
    print("核对: python3 piper_tools/configure_gripper_official.py --check")


if __name__ == "__main__":
    main()
