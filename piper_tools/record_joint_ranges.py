#!/usr/bin/env python3
"""READ-ONLY 关节活动范围记录器 — 手动把各关节拖到最大/最小, 自动记录每臂每关节 [min,max]。

不使能、不发运动、不写固件。只 ConnectPort + GetArmJointMsgs (纯读)。

用法:
  1) 停掉 teleop / 所有 piper ROS 节点 (CAN 保持 UP)。手动失能/拖动各臂(或保持可拖动状态)。
  2) 运行:  python3 record_joint_ranges.py            # 默认记录左右共 4 条臂
            python3 record_joint_ranges.py --cans can_left_mas,can_left_slave   # 只记录左臂主从
  3) 慢慢把每个关节拖到机械最大、再到最小, 来回几次。屏幕实时显示当前值与累计 [min,max]。
  4) Ctrl-C 退出: 打印汇总表 + 存 JSON 到 config/joint_ranges_<时间>.json 作为参考基准。
"""
import argparse
import datetime
import json
import os
import time

from piper_sdk import C_PiperInterface_V2

ALL_CANS = ["can_left_mas", "can_left_slave", "can_right_mas", "can_right_slave"]
JN = ["J1", "J2", "J3", "J4", "J5", "J6"]
OUT_DIR = "/data1/tim/workspace/deepdive_kai0/config"


def read_deg(p):
    js = p.GetArmJointMsgs().joint_state
    return [getattr(js, f"joint_{i}") / 1000.0 for i in range(1, 7)]  # 反馈单位 0.001°


def main():
    ap = argparse.ArgumentParser(description="READ-ONLY 关节范围记录器")
    ap.add_argument("--cans", default=",".join(ALL_CANS),
                    help="逗号分隔的 CAN 接口名 (默认左右 4 臂)")
    ap.add_argument("--hz", type=float, default=10.0, help="采样刷新率")
    ap.add_argument("--out", default=None, help="输出 JSON 路径 (默认 config/joint_ranges_<时间>.json)")
    args = ap.parse_args()

    cans = [c.strip() for c in args.cans.split(",") if c.strip()]
    arms = {}
    for c in cans:
        try:
            p = C_PiperInterface_V2(c); p.ConnectPort()
            arms[c] = p
        except Exception as e:
            print(f"[跳过] {c}: 连接失败 {e}")
    if not arms:
        print("没有可用的臂, 退出。"); return
    time.sleep(0.3)

    # mn/mx[can] = [per-joint] ; None 表示还没读到
    mn = {c: [None] * 6 for c in arms}
    mx = {c: [None] * 6 for c in arms}

    print(f"记录中 ({len(arms)} 臂): {', '.join(arms)}")
    print("把每个关节慢慢拖到最大再到最小, 来回几次。Ctrl-C 结束并保存。\n")
    try:
        while True:
            os.write(1, b"\033[H\033[J")  # 清屏
            print(f"{'arm':16}{'':4}" + "".join(f"{n:>9}" for n in JN))
            for c, p in arms.items():
                d = read_deg(p)
                for i, v in enumerate(d):
                    mn[c][i] = v if mn[c][i] is None else min(mn[c][i], v)
                    mx[c][i] = v if mx[c][i] is None else max(mx[c][i], v)
                print(f"{c:16}{'cur':>4}" + "".join(f"{v:>9.2f}" for v in d))
                print(f"{'':16}{'min':>4}" + "".join(f"{x:>9.2f}" for x in mn[c]))
                print(f"{'':16}{'max':>4}" + "".join(f"{x:>9.2f}" for x in mx[c]))
                print(f"{'':16}{'rng':>4}" + "".join(f"{(b - a):>9.2f}" for a, b in zip(mn[c], mx[c])))
                print()
            print("(拖到极限来回几次让 min/max 收敛; Ctrl-C 保存)")
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass

    # 汇总 + 保存
    print("\n\n===== 关节活动范围汇总 (度) =====")
    result = {}
    for c in arms:
        result[c] = {JN[i]: {"min": round(mn[c][i], 2), "max": round(mx[c][i], 2),
                             "span": round(mx[c][i] - mn[c][i], 2)} for i in range(6)}
        print(f"\n[{c}]")
        print(f"{'':6}" + "".join(f"{n:>9}" for n in JN))
        print(f"{'min':>6}" + "".join(f"{mn[c][i]:>9.2f}" for i in range(6)))
        print(f"{'max':>6}" + "".join(f"{mx[c][i]:>9.2f}" for i in range(6)))
        print(f"{'span':>6}" + "".join(f"{mx[c][i] - mn[c][i]:>9.2f}" for i in range(6)))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or os.path.join(OUT_DIR, f"joint_ranges_{ts}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"recorded_at": ts, "unit": "deg", "ranges": result}, f, indent=2, ensure_ascii=False)
    print(f"\n已保存: {out}")
    print("(纯读, 未做任何写入/运动)")


if __name__ == "__main__":
    main()
