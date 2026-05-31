#!/usr/bin/env python3
# -*-coding:utf8-*-
"""平滑把双臂送到目标关节位姿 —— publish /master/joint_{left,right} (JointState)。

用途: VLA 部署前把臂归到 **demo 起始分布位姿**, 减小 absolute-EE 模型从第 1 帧就 OOD
(见 docs/deployment/inference/xvla_inference_bringup.md §IK调参 / home 对齐)。

工作方式 (ROS2 路径, 非 SDK 直连):
  - 读 /puppet/joint_{left,right} 拿当前关节;
  - current → target 做 cosine 缓入缓出插值, 按 rate 逐步 publish /master/joint_{left,right};
  - arm_reader_node (mode=1) 订阅 /master/* 驱动从臂 → 平滑到位。

前置条件 (重要):
  * autonomy/teleop stack 处于 **OBSERVE** (execute=false), 这样 policy node 不发 /master/*,
    不会和本工具抢话题; arm_reader 仍在跑负责驱动从臂。
  * 不要在 execute=true 或 teleop 主臂在发 /master/* 时运行 —— 会双发冲突。
  * 本工具**会让真机运动**, 默认需交互确认 (--yes 跳过); 全程人工监护。

用法:
  # 双臂归到 A_0423_0527 demo 起始位 (默认 preset), 4s, 保持当前夹爪
  python3 piper_tools/go_to_pose.py
  python3 piper_tools/go_to_pose.py --duration 5 --yes
  # 只看计划不动臂
  python3 piper_tools/go_to_pose.py --dry-run
  # 自定义目标 (6 关节, 度) + 夹爪 (m)
  python3 piper_tools/go_to_pose.py --left -4.6 3.5 -21.8 7 36.1 -11.4 --right 7 3.2 -20.9 -4.1 35.7 4.5
  python3 piper_tools/go_to_pose.py --arm left            # 只动左臂
"""
import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

DEG = math.pi / 180.0

# A_0423_0527 demo 起始位姿 (20 episode 首帧 observation.state 均值; joint rad, gripper m)。
PRESET_LEFT_DEG = [-4.6, 3.5, -21.8, 7.0, 36.1, -11.4]
PRESET_RIGHT_DEG = [7.0, 3.2, -20.9, -4.1, 35.7, 4.5]

# 安全上限: 单关节 current→target 偏差超此 (rad) 直接中止 (防当前读数异常/目标离谱)。
MAX_TOTAL_DELTA_RAD = 100.0 * DEG


class GoToPose(Node):
    def __init__(self, args):
        super().__init__('go_to_pose')
        self.args = args
        self._cur = {'left': None, 'right': None}
        self.create_subscription(JointState, '/puppet/joint_left',
                                 lambda m: self._on_joint('left', m), 10)
        self.create_subscription(JointState, '/puppet/joint_right',
                                 lambda m: self._on_joint('right', m), 10)
        self.pub = {
            'left': self.create_publisher(JointState, '/master/joint_left', 10),
            'right': self.create_publisher(JointState, '/master/joint_right', 10),
        }

    def _on_joint(self, arm, msg):
        if len(msg.position) >= 7:
            self._cur[arm] = list(msg.position[:7])

    def _wait_current(self, arms, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(self._cur[a] is not None for a in arms):
                return True
        return False

    def _publish(self, arm, q6, grip):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = [float(x) for x in q6] + [float(grip)]
        self.pub[arm].publish(msg)

    def run(self):
        a = self.args
        arms = ['left', 'right'] if a.arm == 'both' else [a.arm]

        if not self._wait_current(arms):
            self.get_logger().error('超时未收到 /puppet/joint_* — autonomy/teleop stack 起了吗?')
            return 1

        targets = {}
        for arm in arms:
            cur = self._cur[arm]
            tgt6 = [v * DEG for v in (a.left if arm == 'left' else a.right)]
            grip = cur[6] if a.grip is None else a.grip   # 默认保持当前夹爪, 不意外开合
            targets[arm] = (tgt6, grip)

        # ── 安全检查 + 计划打印 ──
        print('\n=== go_to_pose 计划 ===')
        abort = False
        for arm in arms:
            cur = self._cur[arm]
            tgt6, grip = targets[arm]
            d = [abs(tgt6[i] - cur[i]) for i in range(6)]
            dmax = max(d)
            print(f'[{arm}] 当前 j(deg)={[round(x/DEG,1) for x in cur[:6]]} grip={round(cur[6],3)}')
            print(f'      目标 j(deg)={[round(x/DEG,1) for x in tgt6]} grip={round(grip,3)}')
            print(f'      单关节最大偏差 {dmax/DEG:.1f}°  (各关节 {[round(x/DEG,1) for x in d]})')
            if dmax > MAX_TOTAL_DELTA_RAD and not a.force:
                print(f'      !! 偏差超安全上限 {MAX_TOTAL_DELTA_RAD/DEG:.0f}° → 中止 (确认安全可加 --force)')
                abort = True
        print(f'时长 {a.duration}s @ {a.rate}Hz, cosine 缓入缓出.')
        if abort:
            return 2
        if a.dry_run:
            print('--dry-run: 不发布, 退出.'); return 0
        if not a.yes:
            try:
                if input('臂将运动, 确认? [y/N] ').strip().lower() not in ('y', 'yes'):
                    print('已取消.'); return 0
            except EOFError:
                print('非交互且未加 --yes → 取消.'); return 0

        # ── cosine 缓入缓出插值发布 ──
        n = max(1, int(a.duration * a.rate))
        dt = 1.0 / a.rate
        for k in range(1, n + 1):
            s = 0.5 * (1 - math.cos(math.pi * k / n))  # 0→1, 端点速度=0
            for arm in arms:
                cur = self._cur[arm]
                tgt6, grip = targets[arm]
                q = [cur[i] + (tgt6[i] - cur[i]) * s for i in range(6)]
                g = cur[6] + (grip - cur[6]) * s
                self._publish(arm, q, g)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)
        # 末端多发几帧锁定目标
        for _ in range(int(0.3 * a.rate)):
            for arm in arms:
                tgt6, grip = targets[arm]
                self._publish(arm, tgt6, grip)
            time.sleep(dt)
        print('到位.')
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--left', nargs=6, type=float, default=PRESET_LEFT_DEG,
                   metavar='J', help='左臂 6 关节目标 (度); 默认 A_0423 demo 起始位')
    p.add_argument('--right', nargs=6, type=float, default=PRESET_RIGHT_DEG,
                   metavar='J', help='右臂 6 关节目标 (度); 默认 A_0423 demo 起始位')
    p.add_argument('--arm', choices=['both', 'left', 'right'], default='both')
    p.add_argument('--grip', type=float, default=None, help='夹爪目标 (m); 缺省=保持当前')
    p.add_argument('--duration', type=float, default=4.0, help='运动时长 s (默认 4)')
    p.add_argument('--rate', type=float, default=50.0, help='发布频率 Hz (默认 50)')
    p.add_argument('--yes', action='store_true', help='跳过交互确认')
    p.add_argument('--force', action='store_true', help='跳过单关节偏差安全上限')
    p.add_argument('--dry-run', action='store_true', help='只打印计划, 不发布')
    args = p.parse_args()

    rclpy.init()
    node = GoToPose(args)
    try:
        rc = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()
