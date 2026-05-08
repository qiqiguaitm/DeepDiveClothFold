#!/usr/bin/env python3
"""
交互式 CAN-臂映射校准脚本

用法:
  python3 can_tools/calibrate_can_mapping.py                # 只校准左右映射
  python3 can_tools/calibrate_can_mapping.py --setup-roles  # 先写入 master/slave 角色再校准
  python3 can_tools/calibrate_can_mapping.py --roles-only   # 只写入角色, 跳过映射校准

前置条件: 所有 CAN 接口已激活为 canX (由 setup_can.sh Step 2 完成)

流程:
  0. (可选, --setup-roles) HITL 按角色顺序写入 master/slave
     依次提示:  Master 左 → Master 右 → Slave 左 → Slave 右
     每个角色: 用户晃动对应的臂 → 脚本识别是哪个 iface → 写入
                                                      MasterSlaveConfig(0xFA 或 0xFC)
     这一步完成后左/右/主/从的 iface 映射已经100%确定,
     于是下面的 Phase 1-3 会被整体跳过, 直接进入保存环节.
  1. 枚举所有已激活的 CAN 接口及其 bus-info
  2. 自动分类 slave / master (根据伺服反馈 Hz 和关节值)  [仅非 --setup-roles 走到]
  3. 依次提示用户晃动指定臂, 区分左右                  [仅非 --setup-roles 走到]
  4. 输出校准结果, 保存到 config/pipers.yml 和 activate_can.sh
"""

import argparse
import os
import re
import select
import subprocess
import sys
import time

# ── 配置 ─────────────────────────────────────────────────────────────────────

ROLES = [
    ("left_master", "左 master (示教左臂)"),
    ("left_slave", "左 slave (执行左臂)"),
    ("right_master", "右 master (示教右臂)"),
    ("right_slave", "右 slave (执行右臂)"),
]

SYMBOLIC_NAMES = {
    "left_master": "can_left_mas",
    "left_slave": "can_left_slave",
    "right_master": "can_right_mas",
    "right_slave": "can_right_slave",
}

MOVE_THRESHOLD = 100
DETECT_SECONDS = 8
BITRATE = 1000000

# Piper MasterSlaveConfig linkage_config 字节
# 0xFA = 设置为示教输入臂 (master), 0xFC = 设置为运动输出臂 (slave)
ROLE_MASTER_BYTE = 0xFA
ROLE_SLAVE_BYTE = 0xFC
ROLE_DETECT_SECONDS = 4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # can_tools/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def get_can_interfaces():
    """返回 {接口名: bus_info} 字典, 只返回 UP 状态的接口"""
    out = subprocess.run(
        ["ip", "-br", "link", "show", "type", "can"],
        capture_output=True, text=True, timeout=5,
    )
    result = {}
    for line in out.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        name = parts[0]
        # 只选 UP 的
        if "UP" not in line:
            continue
        try:
            out2 = subprocess.run(
                ["sudo", "ethtool", "-i", name],
                capture_output=True, text=True, timeout=5,
            )
            bus = ""
            for l in out2.stdout.split("\n"):
                if "bus-info" in l:
                    idx = l.index("bus-info:")
                    bus = l[idx + len("bus-info:"):].strip()
            result[name] = bus
        except Exception:
            result[name] = ""
    return result


def check_can_ready(iface):
    """检查接口是否 UP 且 bitrate 已设置 (不依赖 candump, Piper 静止时不主动发帧)"""
    try:
        out = subprocess.run(
            ["ip", "-details", "link", "show", iface],
            capture_output=True, text=True, timeout=5,
        )
        text = out.stdout
        is_up = "state UP" in text
        has_bitrate = "bitrate" in text
        return is_up and has_bitrate
    except Exception:
        return False


def classify_slave_master(interfaces):
    """自动分类 slave/master: slave 有伺服环, 静止时关节值非零; master 无伺服, 静止时全零。
    返回 (slave_list, master_list)"""
    from piper_sdk import C_PiperInterface_V2

    slaves = []
    masters = []

    for iface in interfaces:
        try:
            p = C_PiperInterface_V2(can_name=iface)
            p.ConnectPort()
            time.sleep(1.5)  # 等 SDK 收到反馈帧
            msgs = p.GetArmJointMsgs()
            js = msgs.joint_state
            vals = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
            hz = msgs.Hz
            # slave: 伺服环持续通信, Hz > 0 且关节值不全为零
            if hz > 0 and any(v != 0 for v in vals):
                slaves.append(iface)
            else:
                masters.append(iface)
        except Exception:
            masters.append(iface)  # 连不上的归为 master (保守处理)

    return slaves, masters


# ── 角色配置 (写入臂固件) ────────────────────────────────────────────────────

def set_arm_role(iface, role_byte):
    """通过 MasterSlaveConfig 将 iface 对应的臂设置为 master(0xFA) 或 slave(0xFC)。
    返回 True/False。注意: 角色切换后需断电重启才生效。"""
    from piper_sdk import C_PiperInterface_V2

    try:
        p = C_PiperInterface_V2(can_name=iface)
        p.ConnectPort()
        time.sleep(0.5)
        p.MasterSlaveConfig(role_byte, 0, 0, 0)
        time.sleep(0.3)
        return True
    except Exception as e:
        print(f"  [FAIL] 写入 {iface} 角色失败: {e}")
        return False


def _candump_count(interfaces, seconds, show_progress=False, baseline_rate=None):
    """被动计 candump 帧数 (不向 CAN 发送任何东西, 因此不会触发 SDK 的 SEND_FAILED 日志)。
    返回 {iface: frame_count} 的字典。
    如传入 baseline_rate, 在进度条上显示 delta-vs-baseline。"""
    procs = {}
    for iface in interfaces:
        p = subprocess.Popen(
            ["candump", iface],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        procs[iface] = p

    counts = {iface: 0 for iface in interfaces}
    t0 = time.time()
    last_print = -1
    try:
        while time.time() - t0 < seconds:
            time.sleep(0.15)
            for iface, p in procs.items():
                while select.select([p.stdout], [], [], 0)[0]:
                    line = p.stdout.readline()
                    if line:
                        counts[iface] += 1
                    else:
                        break
            if show_progress:
                elapsed = time.time() - t0
                if int(elapsed) > last_print:
                    last_print = int(elapsed)
                    remaining_s = max(0, seconds - elapsed)
                    parts = []
                    for iface in interfaces:
                        rate = counts[iface] / max(elapsed, 0.1)
                        if baseline_rate is not None:
                            delta = rate - baseline_rate.get(iface, 0)
                            marker = " <<<" if delta > 20 else ""
                            parts.append(f"{iface}={delta:+4.0f}Hz{marker}")
                        else:
                            parts.append(f"{iface}={rate:4.0f}Hz")
                    sys.stdout.write(f"\r  [{remaining_s:4.1f}s] {' | '.join(parts)}   ")
                    sys.stdout.flush()
    finally:
        for p in procs.values():
            p.terminate()
            p.wait()
    if show_progress:
        sys.stdout.write("\n")
    return counts


def _identify_shaken_arm(remaining, interfaces, target_desc=None):
    """识别用户刚刚晃动的臂 —— 混合检测:
      - master (quiet iface, 静止基线 < 50 Hz):
          candump 帧率差分, 晃动后有 joint_ctrl 帧涌出, delta 明显 > 0
      - slave  (chatty iface, 静止基线 > 50 Hz):
          SDK GetArmJointMsgs() 读关节角差分, 因为 slave 伺服回报
          是时间驱动 (200Hz 恒定) —— 帧率本身无法区分静止/晃动, 必须看帧内容
    单次用户晃动在一个统一轮询循环里同时采两种信号, 任一信号超过阈值就算识别成功,
    按各自阈值的 ×倍 取最强者。

    target_desc: HITL 模式下传入当前角色名 (如 "Master 左"), 用于让提示词具体化。
                 不传则退回到泛用提示 "晃动你要配置的任意一臂"。
    返回 iface 或 None (用户取消)。"""
    from piper_sdk import C_PiperInterface_V2

    BASELINE_SECS = 1.0
    CHATTY_BASE_HZ = 50        # 基线 > 此值 → 当作 slave (走 SDK)
    CANDUMP_DELTA_HZ = 20      # master 晃动后帧率至少高出基线 20 Hz
    SDK_JOINT_MDEG = MOVE_THRESHOLD  # slave 关节角变化阈值 (mdeg), 复用 100

    # ── 基线测量 ────────────────────────────────────────────────────────────
    print(f"  基线测量 ({BASELINE_SECS:.0f}s, 请保持所有臂静止)...", end=" ", flush=True)
    baseline = _candump_count(remaining, BASELINE_SECS, show_progress=False)
    base_rate = {i: baseline[i] / BASELINE_SECS for i in remaining}
    print("done")
    quiet = [i for i in remaining if base_rate[i] < CHATTY_BASE_HZ]
    chatty = [i for i in remaining if base_rate[i] >= CHATTY_BASE_HZ]
    for i in remaining:
        label = "slave → 走 SDK 关节角检测" if i in chatty else "master → 走 candump 帧率检测"
        print(f"    {i}: 基线 {base_rate[i]:5.0f} Hz  [{label}]")
    print()

    # ── 为 slave (chatty) 打开 SDK 连接并记录关节基线 ────────────────────────
    # master 上打开 SDK 会触发 "SEND_MESSAGE_FAILED" 错误刷屏, 所以仅在 slave 上打开.
    # 必须放在 try/finally 里保证每次调用结束后 DisconnectPort, 否则每个 role
    # 迭代累计打开的 piper 后台线程 + 周期性 housekeeping 发送会把 CAN 总线拖垮,
    # 进而让后续迭代的 candump 子进程卡住 (观察到在第 3 个角色 "Slave 左"
    # 的基线测量里直接 hang).
    sdk_pipers = {}
    sdk_baseline = {}
    cd_procs = {}
    cd_counts = {iface: 0 for iface in quiet}
    sdk_max_delta = {iface: 0 for iface in chatty}

    try:
        for iface in chatty:
            try:
                p = C_PiperInterface_V2(can_name=iface)
                p.ConnectPort()
                sdk_pipers[iface] = p
            except Exception:
                pass
        if sdk_pipers:
            time.sleep(0.3)  # 等 SDK 收到第一帧反馈
            for iface, p in sdk_pipers.items():
                try:
                    js = p.GetArmJointMsgs().joint_state
                    sdk_baseline[iface] = [js.joint_1, js.joint_2, js.joint_3,
                                            js.joint_4, js.joint_5, js.joint_6]
                except Exception:
                    pass

        # ── 为 master (quiet) 打开 candump 子进程 ────────────────────────────
        for iface in quiet:
            cd_procs[iface] = subprocess.Popen(
                ["candump", iface],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )

        # ── 用户晃动窗口 ────────────────────────────────────────────────────
        if target_desc:
            prompt = f"  按 Enter, 持续晃动【{target_desc}】对应的那只臂, 检测 {ROLE_DETECT_SECONDS}s ..."
        else:
            prompt = f"  按 Enter, 持续晃动你要配置的那只臂, 检测 {ROLE_DETECT_SECONDS}s ..."
        input(prompt)
        t0 = time.time()
        last_print = -1
        while time.time() - t0 < ROLE_DETECT_SECONDS:
            time.sleep(0.15)
            # 读 candump (masters)
            for iface, p in cd_procs.items():
                while select.select([p.stdout], [], [], 0)[0]:
                    if p.stdout.readline():
                        cd_counts[iface] += 1
                    else:
                        break
            # 读 SDK 关节角 (slaves)
            for iface, p in sdk_pipers.items():
                if iface not in sdk_baseline:
                    continue
                try:
                    js = p.GetArmJointMsgs().joint_state
                    cur = [js.joint_1, js.joint_2, js.joint_3,
                           js.joint_4, js.joint_5, js.joint_6]
                    diffs = [abs(a - b) for a, b in zip(cur, sdk_baseline[iface])]
                    md = max(diffs) if diffs else 0
                    if md > sdk_max_delta[iface]:
                        sdk_max_delta[iface] = md
                except Exception:
                    pass
            # 进度条 (每秒一次)
            elapsed = time.time() - t0
            if int(elapsed) > last_print:
                last_print = int(elapsed)
                remaining_s = max(0, ROLE_DETECT_SECONDS - elapsed)
                parts = []
                for iface in remaining:
                    if iface in chatty:
                        d = sdk_max_delta.get(iface, 0)
                        marker = " <<<" if d > SDK_JOINT_MDEG else ""
                        parts.append(f"{iface}={d:5d}md{marker}")
                    else:
                        rate = cd_counts[iface] / max(elapsed, 0.1)
                        delta = rate - base_rate[iface]
                        marker = " <<<" if delta > CANDUMP_DELTA_HZ else ""
                        parts.append(f"{iface}={delta:+5.0f}Hz{marker}")
                sys.stdout.write(f"\r  [{remaining_s:4.1f}s] {' | '.join(parts)}   ")
                sys.stdout.flush()
    finally:
        for p in cd_procs.values():
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        # 关键: 必须关闭 SDK piper 释放后台线程, 否则下次迭代会累计多个 SDK 听众 +
        # housekeeping 写入, 直到 CAN 总线被压垮 / Python 子进程 fork 卡住.
        for iface, p in sdk_pipers.items():
            try:
                p.DisconnectPort()
            except Exception:
                pass
    sys.stdout.write("\n")

    # ── 打分 (每个 iface 算一个 "阈值倍数", 跨 master/slave 可比) ──────────
    scores = {}  # iface -> (ratio, method, raw_value, unit)
    for iface in remaining:
        if iface in chatty:
            d = sdk_max_delta.get(iface, 0)
            scores[iface] = (d / SDK_JOINT_MDEG, "SDK 关节角", d, "mdeg")
        else:
            active_rate = cd_counts.get(iface, 0) / ROLE_DETECT_SECONDS
            delta = active_rate - base_rate[iface]
            scores[iface] = (delta / CANDUMP_DELTA_HZ, "candump 帧率", delta, "Hz")

    print("  检测结果 (按信号强度排序):")
    for iface in sorted(remaining, key=lambda x: scores[x][0], reverse=True):
        ratio, method, raw, unit = scores[iface]
        print(f"    {iface}: {method:10s} delta {raw:+7.1f} {unit} (×{ratio:4.1f} 阈值)")
    print()

    best = max(scores, key=lambda i: scores[i][0])
    if scores[best][0] >= 1.0:
        bus = interfaces.get(best, "?")
        ratio, method, raw, unit = scores[best]
        print(f"  ✓ 识别到: {best} (bus: {bus}, {method} delta {raw:+.1f} {unit})")
        confirm = input("  确认? [Y/n] ").strip().lower()
        if confirm in ("", "y", "yes"):
            return best
        print()

    # ── 自动未达阈值 → 手动回退 ───────────────────────────────────────────
    print(f"  最强信号只到阈值的 {scores[best][0]:.1f}x, 未能识别。")
    print(f"  可能原因: 晃动幅度太小 (slave) / 臂未上电 (master)。")
    while True:
        manual = input(
            f"  手动输入接口名 [候选: {', '.join(remaining)}], 或 [r] 重试, [c] 取消: "
        ).strip()
        if manual.lower() == "r":
            return _identify_shaken_arm(remaining, interfaces, target_desc)
        if manual.lower() == "c":
            return None
        if manual in remaining:
            return manual
        print(f"  无效: '{manual}'")


# 角色分配顺序 (HITL 按此顺序提示): 先两个 master, 再两个 slave.
# 每项: (mapping_key, 显示名, 目标角色字节)
ROLE_ASSIGN_SEQ = [
    ("left_master",  "Master 左 (示教左臂)",  ROLE_MASTER_BYTE),
    ("right_master", "Master 右 (示教右臂)",  ROLE_MASTER_BYTE),
    ("left_slave",   "Slave  左 (执行左臂)",  ROLE_SLAVE_BYTE),
    ("right_slave",  "Slave  右 (执行右臂)",  ROLE_SLAVE_BYTE),
]


def role_setup_wizard(interfaces):
    """交互式按角色顺序配置 master/slave 并写入臂固件。

    流程 (HITL, 按 ROLE_ASSIGN_SEQ 顺序):
      for 每个角色 (master_L → master_R → slave_L → slave_R):
          提示用户晃动"这只臂要做 <角色>"
          识别被晃动的 iface (候选 = remaining)
          记录 role_key → iface, 累加到待写入队列
      显示汇总 → 确认 → 依次 MasterSlaveConfig 写入.

    副产物: 在这一步就确定了完整 left/right_master/slave 的 iface 映射,
    main() 可据此跳过 Phase 1-3.

    返回 (success: bool, role_mapping: dict[role_key, iface]).
    只有 success=True 且 mapping 含全部 4 项时, main() 才会跳过后续校准阶段."""
    print("=" * 60)
    print("  阶段 0: 按角色顺序配置 master/slave (HITL)")
    print("=" * 60)
    print()
    print("  按顺序提示 4 个角色, 每次:")
    print("    1) 指定将成为【某角色】的物理臂, 持续晃动它 4 秒")
    print("    2) 脚本识别它对应的 CAN 接口 (candump 帧率 + SDK 关节角混合检测)")
    print("    3) 确认后下发 MasterSlaveConfig 写入固件")
    print()
    print("  注意:")
    print("    - master → slave 切换后必须给该臂【断电重启】才生效 (SDK 文档)")
    print("    - slave → master 切换无需重启")
    print()

    remaining = list(interfaces.keys())
    role_mapping: dict = {}   # role_key -> iface
    pending_writes: list = [] # [(iface, byte, role_desc)] 按顺序累计, 最后一起写

    for role_key, desc, role_byte in ROLE_ASSIGN_SEQ:
        if not remaining:
            print(f"  接口已分完, 无法继续分配 {role_key}")
            break

        print("-" * 60)
        print(f"  下一个角色: 【{desc}】")
        if role_mapping:
            assigned = ", ".join(f"{k}={v}" for k, v in role_mapping.items())
            print(f"  已分配: {assigned}")
        print(f"  剩余接口: {', '.join(remaining)}")
        print()
        print(f"  → 请【持续晃动】你希望设为 {desc} 的那只臂")
        print()

        iface = _identify_shaken_arm(remaining, interfaces, target_desc=desc)
        if iface is None:
            ans = input(
                f"  未识别到. [s]=跳过 {role_key} 继续下一角色, [q]=退出整个角色设置: "
            ).strip().lower()
            if ans == "s":
                print()
                continue
            # 否则退出整个向导
            print("  已退出.")
            print()
            return False, {}

        role_mapping[role_key] = iface
        pending_writes.append((iface, role_byte, desc))
        remaining.remove(iface)
        role_str = "MASTER (0xFA)" if role_byte == ROLE_MASTER_BYTE else "SLAVE (0xFC)"
        print(f"  ✓ 记录: {role_key} = {iface}  → 将写入 {role_str}")
        print()

    if not pending_writes:
        print("  未分配任何角色, 跳过写入.")
        print()
        return False, {}

    # ── 汇总 + 确认 ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  即将写入以下角色:")
    for role_key, iface in role_mapping.items():
        byte = next(b for (i, b, _) in pending_writes if i == iface)
        byte_str = "MASTER (0xFA)" if byte == ROLE_MASTER_BYTE else "SLAVE (0xFC)"
        bus = interfaces.get(iface, "?")
        print(f"    {role_key:15s} = {iface:8s} (bus: {bus:12s}) → {byte_str}")
    print("=" * 60)
    confirm = input("  确认写入? [y/N] ").strip().lower()
    if confirm not in ("y", "yes"):
        print("  已取消, 不写入.")
        print()
        return False, {}

    # ── 写入 ────────────────────────────────────────────────────────────────
    print()
    ok_count = 0
    switched_to_slave = False
    for iface, byte, desc in pending_writes:
        role_str = "MASTER" if byte == ROLE_MASTER_BYTE else "SLAVE"
        print(f"  写入 {iface} → {role_str} (0x{byte:02X}) [{desc}] ...",
              end=" ", flush=True)
        if set_arm_role(iface, byte):
            print("OK")
            ok_count += 1
            if byte == ROLE_SLAVE_BYTE:
                switched_to_slave = True
        else:
            print("FAIL")

    print()
    print(f"  完成 {ok_count}/{len(pending_writes)} 个角色写入.")
    if switched_to_slave:
        print()
        print("  ⚠ 有臂被设为 SLAVE (0xFC). 如果它之前是 MASTER, 必须【断电重启】该臂,")
        print("    否则伺服环不会启动 (SDK 文档明确要求). slave→master 无需重启.")
        input("  处理完成后按 Enter 继续...")
    print()

    # 只有完整 4 个角色都分配了, 才返回可用于跳过 Phase 1-3 的 mapping
    complete = len(role_mapping) == len(ROLE_ASSIGN_SEQ)
    return complete, role_mapping


def detect_moving_sdk(interfaces, seconds):
    """用 SDK 读关节角度变化量, 检测哪个 slave 在动。返回 (best_iface, deltas_dict)"""
    from piper_sdk import C_PiperInterface_V2

    pipers = {}
    for iface in interfaces:
        try:
            p = C_PiperInterface_V2(can_name=iface)
            p.ConnectPort()
            pipers[iface] = p
        except Exception:
            pass

    if not pipers:
        return None, {}

    time.sleep(1)

    def read(iface):
        msgs = pipers[iface].GetArmJointMsgs()
        js = msgs.joint_state
        return [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]

    baselines = {iface: read(iface) for iface in pipers}
    max_deltas = {iface: 0 for iface in pipers}
    t0 = time.time()
    last_print = 0

    while time.time() - t0 < seconds:
        time.sleep(0.2)
        elapsed = time.time() - t0

        for iface in pipers:
            cur = read(iface)
            base = baselines[iface]
            if cur and base:
                diffs = [abs(a - b) for a, b in zip(cur, base)]
                md = max(diffs)
                if md > max_deltas[iface]:
                    max_deltas[iface] = md

        if int(elapsed) > last_print:
            last_print = int(elapsed)
            remaining = seconds - int(elapsed)
            parts = []
            for iface in pipers:
                marker = " <<<" if max_deltas[iface] > MOVE_THRESHOLD else ""
                parts.append(f"{iface}={max_deltas[iface]:5d}{marker}")
            sys.stdout.write(f"\r  [{remaining:2d}s] {' | '.join(parts)}   ")
            sys.stdout.flush()

    sys.stdout.write("\n")

    best = max(max_deltas, key=max_deltas.get)
    return best if max_deltas[best] > MOVE_THRESHOLD else None, max_deltas


def detect_moving_candump(interfaces, seconds):
    """用 candump 计帧数, 检测哪个 master 在动。master 静止时无帧, 晃动时有帧。
    返回 (best_iface, frame_counts_dict)"""
    import signal

    procs = {}
    for iface in interfaces:
        # candump 持续输出, 每行一帧
        p = subprocess.Popen(
            ["candump", iface],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        procs[iface] = p

    t0 = time.time()
    last_print = 0
    counts = {iface: 0 for iface in interfaces}

    try:
        while time.time() - t0 < seconds:
            time.sleep(0.3)
            elapsed = time.time() - t0

            # 非阻塞读取已有输出, 统计行数
            for iface, p in procs.items():
                while select.select([p.stdout], [], [], 0)[0]:
                    line = p.stdout.readline()
                    if line:
                        counts[iface] += 1
                    else:
                        break

            if int(elapsed) > last_print:
                last_print = int(elapsed)
                remaining = seconds - int(elapsed)
                parts = []
                for iface in interfaces:
                    marker = " <<<" if counts[iface] > 10 else ""
                    parts.append(f"{iface}={counts[iface]:5d}{marker}")
                sys.stdout.write(f"\r  [{remaining:2d}s] {' | '.join(parts)}   ")
                sys.stdout.flush()
    finally:
        for p in procs.values():
            p.terminate()
            p.wait()

    sys.stdout.write("\n")

    if not counts:
        return None, counts
    best = max(counts, key=counts.get)
    return best if counts[best] > 10 else None, counts


# ── 配置写入 ─────────────────────────────────────────────────────────────────

def write_pipers_yml(mapping, bus_infos):
    """更新 config/pipers.yml"""
    path = os.path.join(PROJECT_ROOT, "config", "pipers.yml")

    lines = [
        "# Piper 双臂配置 — sim01 部署",
        f"# 通过 calibrate_can_mapping.py 校准, {time.strftime('%Y-%m-%d %H:%M')}",
        "#",
        "# 启动前运行: bash can_tools/setup_can.sh --quick",
        "",
        "arms:",
    ]

    for role, desc in ROLES:
        if role not in mapping:
            lines.append(f"  # {role}: 未映射 (跳过)")
            lines.append("")
            continue
        iface = mapping[role]
        bus = bus_infos.get(iface, "")
        symbolic = SYMBOLIC_NAMES[role]
        mode = 0 if "master" in role else 1
        side = "左" if "left" in role else "右"
        kind = "master" if "master" in role else "slave"
        topic_prefix = "/master" if "master" in role else "/puppet"
        topic_side = "left" if "left" in role else "right"

        lines += [
            f"  {role}:",
            f"    can_physical: {iface}",
            f"    can_symbolic: {symbolic}",
            f'    usb_bus_info: "{bus}"',
            f"    role: {side}臂 ({kind})",
            f"    mode: {mode}",
            f"    dof: 6",
            f"    feedback_hz: 200",
            f"    can_bitrate: {BITRATE}",
            f"    ros2_topic_joint: {topic_prefix}/joint_{topic_side}",
            "",
        ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  已写入 {path}")


def write_activate_script(mapping, bus_infos):
    """重写 can_tools/activate_can.sh"""
    path = os.path.join(SCRIPT_DIR, "activate_can.sh")
    if not os.path.exists(path):
        print(f"  {path} 不存在, 跳过")
        return

    with open(path) as f:
        content = f.read()

    # 构建新的映射条目
    slave_entries = []
    master_entries = []
    for role, _ in ROLES:
        if role not in mapping:
            continue
        bus = bus_infos.get(mapping[role], "")
        symbolic = SYMBOLIC_NAMES[role]
        entry = f'    "{bus}:{symbolic}"'
        if "slave" in role:
            slave_entries.append(entry)
        else:
            master_entries.append(entry)

    # 替换 SLAVE_MAPPINGS 块
    slave_body = ('\n' + '\n'.join(slave_entries) + '\n') if slave_entries else ''
    content = re.sub(
        r'SLAVE_MAPPINGS=\([^)]*\)',
        f'SLAVE_MAPPINGS=({slave_body})',
        content,
    )
    # 替换 MASTER_MAPPINGS 块
    master_body = ('\n' + '\n'.join(master_entries) + '\n') if master_entries else ''
    content = re.sub(
        r'MASTER_MAPPINGS=\([^)]*\)',
        f'MASTER_MAPPINGS=({master_body})',
        content,
    )

    # 更新头部注释
    comment_lines = []
    for role, desc in ROLES:
        if role not in mapping:
            comment_lines.append(f"#   (未映射) → {SYMBOLIC_NAMES[role]}  ({desc})")
            continue
        bus = bus_infos.get(mapping[role], "?")
        symbolic = SYMBOLIC_NAMES[role]
        comment_lines.append(f"#   {bus} → {symbolic}  ({desc})")
    comment_block = "\n".join(comment_lines)
    content = re.sub(
        r'# sim01 bus-info \(.*?\n(?:#   .*\n)+',
        f"# sim01 bus-info ({time.strftime('%Y-%m-%d')} calibrate_can_mapping.py 校准):\n{comment_block}\n",
        content,
    )

    with open(path, "w") as f:
        f.write(content)
    print(f"  已更新 {path}")


def write_activate_can_arms(mapping, bus_infos):
    """更新 kai0 下的 activate_can_arms.sh"""
    path = os.path.join(
        PROJECT_ROOT, "kai0", "train_deploy_alignment", "dagger", "agilex",
        "activate_can_arms.sh",
    )
    if not os.path.exists(path):
        return

    lines = [
        "#!/bin/bash",
        "# Activate all four USB-CAN interfaces for dual master + dual slave.",
        f"# 由 calibrate_can_mapping.py 自动生成, {time.strftime('%Y-%m-%d %H:%M')}",
        "#",
        "# sim01 bus-info 映射:",
    ]
    for role, desc in ROLES:
        if role not in mapping:
            lines.append(f"#   (未映射) → {SYMBOLIC_NAMES[role]} ({desc})")
            continue
        bus = bus_infos.get(mapping[role], "?")
        lines.append(f"#   {bus} → {SYMBOLIC_NAMES[role]} ({desc})")
    lines.append("")

    for role, _ in ROLES:
        if role not in mapping:
            lines.append(f"# 跳过 {SYMBOLIC_NAMES[role]} (未映射)")
            continue
        bus = bus_infos.get(mapping[role], "")
        symbolic = SYMBOLIC_NAMES[role]
        lines.append(f'bash ./can_activate.sh {symbolic:20s} {BITRATE} "{bus}"')

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  已更新 {path}")


def _manual_pick(candidates, desc):
    """手动选择接口"""
    print(f"  可选接口: {', '.join(candidates)}")
    manual = input(f"  请手动输入 【{desc}】 对应的接口名 (或 Enter 跳过): ").strip()
    if manual in candidates:
        print(f"  已绑定: {manual}")
        return manual
    if manual:
        print(f"  无效输入 '{manual}'")
    return None


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CAN-臂映射交互式校准 + 可选的 master/slave 角色写入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--setup-roles",
        action="store_true",
        help="在映射校准之前, 先交互式写入 master/slave 角色到臂固件",
    )
    parser.add_argument(
        "--roles-only",
        action="store_true",
        help="只做角色写入, 跳过左右映射校准 (隐含 --setup-roles)",
    )
    args = parser.parse_args()

    if args.roles_only:
        args.setup_roles = True

    print("=" * 60)
    print("  CAN-臂映射交互式校准")
    if args.setup_roles:
        print("  (含 master/slave 角色写入)")
    print("=" * 60)
    print()

    # 1. 枚举接口
    all_interfaces = get_can_interfaces()
    if not all_interfaces:
        print("未检测到已激活的 CAN 接口。")
        print("请先运行: bash can_tools/setup_can.sh")
        sys.exit(1)

    print(f"检测到 {len(all_interfaces)} 个活跃 CAN 接口:")
    not_ready = []
    for iface, bus in all_interfaces.items():
        ready = check_can_ready(iface)
        status = "UP ✓" if ready else "异常 ✗"
        if not ready:
            not_ready.append(iface)
        print(f"  {iface:10s}  bus={bus:15s}  {status}")
    print()

    if not_ready:
        print(f"警告: {', '.join(not_ready)} 接口状态异常 (未 UP 或 bitrate 未设置)。")
        cont = input("是否继续校准? [Y/n] ").strip().lower()
        if cont not in ("", "y", "yes"):
            sys.exit(0)
        print()

    # ══ 阶段 0: (可选) 按角色顺序写入 master/slave ═══════════════════════════
    # 向导直接产出完整的 role_mapping (left_master/right_master/left_slave/right_slave → iface),
    # 如果全 4 项都有, 下面的 Phase 1-3 就可以整体跳过.
    preassigned_mapping: dict = {}
    if args.setup_roles:
        wizard_complete, wizard_mapping = role_setup_wizard(all_interfaces)

        if args.roles_only:
            # --roles-only 仍然把映射保存到 pipers.yml / activate_can.sh, 便于后续 --quick 直接激活
            if wizard_mapping:
                print()
                print("保存角色向导的映射到配置文件 ...")
                write_pipers_yml(wizard_mapping, all_interfaces)
                write_activate_script(wizard_mapping, all_interfaces)
                write_activate_can_arms(wizard_mapping, all_interfaces)
            print("=" * 60)
            print("  --roles-only 已完成, 退出 (未执行映射校准)")
            print("  如需完整映射校准, 请重跑: python3 calibrate_can_mapping.py")
            print("=" * 60)
            sys.exit(0)

        # 角色可能改动, 重新枚举接口 (UP 状态、bus-info 通常不变, 但保险起见)
        all_interfaces = get_can_interfaces()
        if not all_interfaces:
            print("角色写入后未检测到 CAN 接口, 退出。")
            sys.exit(1)

        if wizard_complete:
            preassigned_mapping = wizard_mapping

    # ══ 跳过 Phase 1-3 的快速路径 ════════════════════════════════════════════
    # HITL 向导里用户已经按 master_L / master_R / slave_L / slave_R 顺序亲自指定了每个臂,
    # 即左右映射已经100%由人确认. Phase 1 (slave/master 分类) 和 Phase 2/3 (左右) 都多余了.
    if preassigned_mapping:
        print("=" * 60)
        print("  使用角色向导的映射, 跳过 Phase 1-3")
        print("=" * 60)
        print()
        print(f"  {'角色':<18s} {'接口':<10s} {'Bus-Info':<15s} {'符号名':<20s}")
        print(f"  {'─'*16:<18s} {'─'*8:<10s} {'─'*13:<15s} {'─'*18:<20s}")
        for role, desc in ROLES:
            iface = preassigned_mapping.get(role)
            if iface:
                bus = all_interfaces.get(iface, "?")
                symbolic = SYMBOLIC_NAMES[role]
                print(f"  {desc:<16s} {iface:<10s} {bus:<15s} {symbolic:<20s}")
            else:
                print(f"  {desc:<16s} {'—':<10s} {'—':<15s} {'未分配':<20s}")
        print()
        save = input("是否保存到配置文件? [Y/n] ").strip().lower()
        if save in ("", "y", "yes"):
            print()
            write_pipers_yml(preassigned_mapping, all_interfaces)
            write_activate_script(preassigned_mapping, all_interfaces)
            write_activate_can_arms(preassigned_mapping, all_interfaces)
            print()
            print("配置已保存.")
        else:
            print("未保存.")
        sys.exit(0)

    # ══ 阶段 1: 自动分类 slave / master ══════════════════════════════════════
    print("=" * 60)
    print("  阶段 1: 自动识别 slave / master")
    print("=" * 60)
    print()
    print("  slave 臂有伺服环, 静止时持续通信; master 臂无伺服, 静止时无数据。")
    print("  正在连接所有接口并检测...")
    print()

    iface_list = list(all_interfaces.keys())
    slaves, masters = classify_slave_master(iface_list)

    print(f"  Slave  接口 ({len(slaves)}): {', '.join(slaves) if slaves else '无'}")
    print(f"  Master 接口 ({len(masters)}): {', '.join(masters) if masters else '无'}")
    print()

    if len(slaves) != 2 or len(masters) != 2:
        print(f"  预期 2 slave + 2 master, 实际 {len(slaves)} slave + {len(masters)} master")
        print("  自动分类结果异常, 将回退到手动模式。")
        # 回退: 全部手动
        slaves = []
        masters = iface_list[:]
    else:
        confirm = input("  分类正确? [Y/n] ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("  请手动指定。")
            print(f"  所有接口: {', '.join(iface_list)}")
            s = input("  输入两个 slave 接口名 (空格分隔): ").strip().split()
            if len(s) == 2 and all(x in iface_list for x in s):
                slaves = s
                masters = [x for x in iface_list if x not in slaves]
            else:
                print("  输入无效, 退出。")
                sys.exit(1)
    print()

    mapping = {}
    # slave 和 master 各自需要区分左右
    SLAVE_ROLES = [r for r in ROLES if "slave" in r[0]]   # left_slave, right_slave
    MASTER_ROLES = [r for r in ROLES if "master" in r[0]]  # left_master, right_master

    # ══ 阶段 2: 区分 slave 左右 (SDK 关节值检测) ═════════════════════════════
    if len(slaves) >= 2:
        print("=" * 60)
        print("  阶段 2: 区分 slave 左右 (晃动 slave 臂)")
        print("=" * 60)
        print()

        remaining_slaves = slaves[:]
        for role, desc in SLAVE_ROLES:
            if len(remaining_slaves) == 1:
                last = remaining_slaves[0]
                print(f"  仅剩一个 slave, 自动绑定: 【{desc}】 = {last}")
                mapping[role] = last
                remaining_slaves.remove(last)
                continue

            print("-" * 60)
            print(f"  请晃动 【{desc}】")
            print(f"  候选: {', '.join(remaining_slaves)}")
            print(f"  检测时间: {DETECT_SECONDS} 秒 (SDK 关节值变化)")
            print()
            input("  准备好后按 Enter 开始检测...")

            detected, deltas = detect_moving_sdk(remaining_slaves, DETECT_SECONDS)

            if detected:
                bus = all_interfaces.get(detected, "?")
                print(f"  ✓ 检测到 【{desc}】 → {detected} (bus: {bus}, delta: {deltas[detected]})")
                confirm = input("  确认? [Y/n] ").strip().lower()
                if confirm in ("", "y", "yes"):
                    mapping[role] = detected
                    remaining_slaves.remove(detected)
                else:
                    manual = _manual_pick(remaining_slaves, desc)
                    if manual:
                        mapping[role] = manual
                        remaining_slaves.remove(manual)
            else:
                print("  ✗ 未检测到明显运动")
                manual = _manual_pick(remaining_slaves, desc)
                if manual:
                    mapping[role] = manual
                    remaining_slaves.remove(manual)
            print()
    elif len(slaves) == 1:
        # 只有 1 个 slave, 需要手动指定是左还是右
        print(f"  只有 1 个 slave: {slaves[0]}")
        side = input("  它是 左slave 还是 右slave? [l/r] ").strip().lower()
        if side in ("l", "left"):
            mapping["left_slave"] = slaves[0]
        else:
            mapping["right_slave"] = slaves[0]
        print()

    # ══ 阶段 3: 区分 master 左右 (candump 计帧检测) ══════════════════════════
    if len(masters) >= 2:
        print("=" * 60)
        print("  阶段 3: 区分 master 左右 (晃动 master 臂)")
        print("  master 无伺服, 使用 candump 计帧检测")
        print("=" * 60)
        print()

        remaining_masters = masters[:]
        for role, desc in MASTER_ROLES:
            if len(remaining_masters) == 1:
                last = remaining_masters[0]
                print(f"  仅剩一个 master, 自动绑定: 【{desc}】 = {last}")
                mapping[role] = last
                remaining_masters.remove(last)
                continue

            print("-" * 60)
            print(f"  请晃动 【{desc}】 (持续晃动, 不要停)")
            print(f"  候选: {', '.join(remaining_masters)}")
            print(f"  检测时间: {DETECT_SECONDS} 秒 (candump 帧计数)")
            print()
            input("  准备好后按 Enter 开始检测...")

            detected, counts = detect_moving_candump(remaining_masters, DETECT_SECONDS)

            if detected:
                bus = all_interfaces.get(detected, "?")
                print(f"  ✓ 检测到 【{desc}】 → {detected} (bus: {bus}, frames: {counts[detected]})")
                confirm = input("  确认? [Y/n] ").strip().lower()
                if confirm in ("", "y", "yes"):
                    mapping[role] = detected
                    remaining_masters.remove(detected)
                else:
                    manual = _manual_pick(remaining_masters, desc)
                    if manual:
                        mapping[role] = manual
                        remaining_masters.remove(manual)
            else:
                print("  ✗ 未检测到 CAN 帧 (确保在晃动臂)")
                manual = _manual_pick(remaining_masters, desc)
                if manual:
                    mapping[role] = manual
                    remaining_masters.remove(manual)
            print()
    elif len(masters) == 1:
        print(f"  只有 1 个 master: {masters[0]}")
        side = input("  它是 左master 还是 右master? [l/r] ").strip().lower()
        if side in ("l", "left"):
            mapping["left_master"] = masters[0]
        else:
            mapping["right_master"] = masters[0]
        print()

    # ══ 阶段 4: 手动兜底 (剩余角色 ↔ 剩余接口) ═══════════════════════════════
    # 自动检测可能漏掉某些臂 (例如 slave 没上电导致阶段 1 全部归为 master,
    # 阶段 3 又只识别了用户晃动过的那几个). 这里给一次手动绑定的机会.
    unmapped_roles = [(r, d) for r, d in ROLES if r not in mapping]
    used_ifaces = set(mapping.values())
    leftover_ifaces = [i for i in iface_list if i not in used_ifaces]

    if unmapped_roles and leftover_ifaces:
        print("=" * 60)
        print("  阶段 4: 手动兜底")
        print("=" * 60)
        print()
        print(f"  未映射角色 ({len(unmapped_roles)}): "
              + ", ".join(d for _, d in unmapped_roles))
        print(f"  剩余接口   ({len(leftover_ifaces)}): "
              + ", ".join(leftover_ifaces))
        print()
        print("  提示: 自动分类没认出 slave 通常是 slave 臂没上电、")
        print("        或 master/slave 角色字节没写好。")
        print("        如果你确定剩余接口对应哪个臂, 可以在这里手动指定;")
        print("        否则按 Enter 跳过, 等修好硬件再重跑。")
        print()

        for role, desc in unmapped_roles:
            if not leftover_ifaces:
                break
            print("-" * 60)
            print(f"  【{desc}】")
            print(f"  候选: {', '.join(leftover_ifaces)}")
            choice = input(f"  请输入接口名 (Enter 跳过): ").strip()
            if not choice:
                print("  跳过。")
                print()
                continue
            if choice not in leftover_ifaces:
                print(f"  无效输入 '{choice}', 跳过。")
                print()
                continue
            mapping[role] = choice
            leftover_ifaces.remove(choice)
            bus = all_interfaces.get(choice, "?")
            print(f"  已绑定: 【{desc}】 → {choice} (bus: {bus})")
            print()

    # 3. 显示结果
    print("=" * 60)
    print("  校准结果")
    print("=" * 60)
    print()
    print(f"  {'角色':<20s} {'接口':<10s} {'Bus-Info':<15s} {'符号名':<20s}")
    print(f"  {'─'*18:<20s} {'─'*8:<10s} {'─'*13:<15s} {'─'*18:<20s}")
    for role, desc in ROLES:
        if role in mapping:
            iface = mapping[role]
            bus = all_interfaces.get(iface, "?")
            symbolic = SYMBOLIC_NAMES[role]
            print(f"  {desc:<20s} {iface:<10s} {bus:<15s} {symbolic:<20s}")
        else:
            print(f"  {desc:<20s} {'—':<10s} {'—':<15s} {'未绑定':<20s}")
    print()

    # 4. 保存
    mapped_count = len(mapping)
    if mapped_count == 0:
        print("未完成任何映射, 退出。")
        sys.exit(1)

    if mapped_count < 4:
        print(f"注意: 只完成了 {mapped_count}/4 个映射。")

    save = input("是否保存到配置文件? [Y/n] ").strip().lower()
    if save in ("", "y", "yes"):
        print()
        write_pipers_yml(mapping, all_interfaces)
        write_activate_script(mapping, all_interfaces)
        write_activate_can_arms(mapping, all_interfaces)
        print()
        print("配置已保存。")
    else:
        print("未保存。")
        sys.exit(0)


if __name__ == "__main__":
    main()
