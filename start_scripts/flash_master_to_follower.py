#!/usr/bin/env python3
"""Flash master arm firmware role from 0xFA (leader) to 0xFC (follower).

After running this script:
1. Physically POWER CYCLE the master arm (unplug 24V from arm side, wait 10s, replug)
2. After reboot, master will accept CAN_CTRL mode + JointCtrl commands
3. To switch back to leader role: re-run setup_can.sh --roles-only
"""
import sys, time
from piper_sdk import C_PiperInterface_V2


def main():
    if len(sys.argv) < 2:
        print("Usage: flash_master_to_follower.py <can_port> [<can_port2> ...]")
        print("Example: flash_master_to_follower.py can_left_mas can_right_mas")
        return 1

    for port in sys.argv[1:]:
        print(f"\n=== {port}: writing MasterSlaveConfig(0xFC) — follower role ===")
        arm = C_PiperInterface_V2(can_name=port, judge_flag=False)
        arm.ConnectPort()
        time.sleep(0.5)

        # Send 5 times for robustness (firmware NVRAM write)
        for i in range(5):
            arm.MasterSlaveConfig(0xFC, 0x00, 0x00, 0x00)
            print(f"  write {i+1}/5 sent")
            time.sleep(0.3)
        time.sleep(1.5)
        print(f"  ✓ {port}: 0xFC written to firmware NVRAM")

    print("\n" + "=" * 60)
    print("NEXT STEPS (physical):")
    print("  1. UNPLUG 24V from master arm (at arm side, not adapter)")
    print("  2. WAIT 10 seconds")
    print("  3. RE-PLUG 24V")
    print("  4. WAIT 5 seconds for firmware boot")
    print("  5. Run /tmp/check_master_alive.sh — should now see frames flowing")
    print("  6. Then run test_master_servo.py — JointCtrl should work")
    print("=" * 60)
    print("\nNote: After this flash, master will no longer drag-teach.")
    print("To restore drag-teach: bash piper_tools/setup_can.sh --roles-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
