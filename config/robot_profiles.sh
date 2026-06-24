#!/usr/bin/env bash
# Robot/machine profile resolver for kai0 deployment scripts.
#
# Machine selection precedence:
#   1. VIS_ROBOT_ID
#   2. KAI0_MACHINE_ID
#   3. hostname -s
#
# Profiles:
#   visrobot02/agilex: current robot, two shared CAN buses (left/right)
#   visrobot01/sim01: original four-CAN robot

resolve_kai0_machine_id() {
    if [[ -n "${VIS_ROBOT_ID:-}" ]]; then
        echo "$VIS_ROBOT_ID"
    elif [[ -n "${KAI0_MACHINE_ID:-}" ]]; then
        echo "$KAI0_MACHINE_ID"
    else
        hostname -s 2>/dev/null || echo "unknown"
    fi
}

apply_kai0_robot_profile() {
    local task="${1:-teleop}"
    local machine
    machine="$(resolve_kai0_machine_id)"
    export KAI0_MACHINE_ID="$machine"

    case "$machine" in
        visrobot02|agilex)
            export KAI0_ROBOT_PROFILE="visrobot02"
            export KAI0_DEFAULT_CAN_TOPOLOGY="2can"
            export KAI0_EXPECTED_CAN_IFACES_DEFAULT="can_left_mas can_right_mas"
            export KAI0_CAN_ACTIVATE_ARGS_DEFAULT="--two-can"
            export KAI0_TELEOP_LAUNCH_DEFAULT="teleop_2can_launch.py"
            export KAI0_READONLY_LAUNCH_DEFAULT="teleop_2can_readonly_launch.py"
            export KAI0_DEFAULT_TELEOP_MODE="readonly"
            export KAI0_LEFT_SHARED_BUS_INFO="${KAI0_LEFT_SHARED_BUS_INFO:-1-13:1.0}"
            export KAI0_RIGHT_SHARED_BUS_INFO="${KAI0_RIGHT_SHARED_BUS_INFO:-1-12:1.0}"
            export KAI0_LEFT_MASTER_BUS_INFO="${KAI0_LEFT_MASTER_BUS_INFO:-$KAI0_LEFT_SHARED_BUS_INFO}"
            export KAI0_RIGHT_MASTER_BUS_INFO="${KAI0_RIGHT_MASTER_BUS_INFO:-$KAI0_RIGHT_SHARED_BUS_INFO}"
            export KAI0_LEFT_SLAVE_BUS_INFO="${KAI0_LEFT_SLAVE_BUS_INFO:-1-1:1.0}"
            ;;
        visrobot01|sim01)
            export KAI0_ROBOT_PROFILE="visrobot01"
            export KAI0_DEFAULT_CAN_TOPOLOGY="4can"
            export KAI0_EXPECTED_CAN_IFACES_DEFAULT="can_left_mas can_left_slave can_right_mas can_right_slave"
            export KAI0_CAN_ACTIVATE_ARGS_DEFAULT="--four-can"
            export KAI0_TELEOP_LAUNCH_DEFAULT="teleop_launch.py"
            export KAI0_READONLY_LAUNCH_DEFAULT=""
            export KAI0_DEFAULT_TELEOP_MODE="active"
            export KAI0_LEFT_MASTER_BUS_INFO="${KAI0_LEFT_MASTER_BUS_INFO:-3-2.2.2:1.0}"
            export KAI0_LEFT_SLAVE_BUS_INFO="${KAI0_LEFT_SLAVE_BUS_INFO:-3-2.2.1:1.0}"
            export KAI0_RIGHT_MASTER_BUS_INFO="${KAI0_RIGHT_MASTER_BUS_INFO:-3-2.2.3:1.0}"
            export KAI0_RIGHT_SLAVE_BUS_INFO="${KAI0_RIGHT_SLAVE_BUS_INFO:-3-2.2.4:1.0}"
            ;;
        *)
            export KAI0_ROBOT_PROFILE="$machine"
            export KAI0_DEFAULT_CAN_TOPOLOGY="auto"
            export KAI0_EXPECTED_CAN_IFACES_DEFAULT=""
            export KAI0_CAN_ACTIVATE_ARGS_DEFAULT=""
            export KAI0_TELEOP_LAUNCH_DEFAULT=""
            export KAI0_READONLY_LAUNCH_DEFAULT=""
            export KAI0_DEFAULT_TELEOP_MODE="active"
            ;;
    esac

    export KAI0_CAN_TOPOLOGY="${KAI0_CAN_TOPOLOGY:-$KAI0_DEFAULT_CAN_TOPOLOGY}"
    export KAI0_EXPECTED_CAN_IFACES="${KAI0_EXPECTED_CAN_IFACES:-$KAI0_EXPECTED_CAN_IFACES_DEFAULT}"
    export KAI0_CAN_ACTIVATE_ARGS="${KAI0_CAN_ACTIVATE_ARGS:-$KAI0_CAN_ACTIVATE_ARGS_DEFAULT}"

    case "$task" in
        data_collect|collect)
            # On visrobot02, master/slave roles live in Piper firmware and
            # 0x470 is broadcast on the shared CAN bus. Data collection must
            # therefore be readonly by default and must not rewrite roles.
            export KAI0_TELEOP_MODE="${KAI0_TELEOP_MODE:-$KAI0_DEFAULT_TELEOP_MODE}"
            ;;
        teleop|*)
            export KAI0_TELEOP_MODE="${KAI0_TELEOP_MODE:-$KAI0_DEFAULT_TELEOP_MODE}"
            ;;
    esac
}
