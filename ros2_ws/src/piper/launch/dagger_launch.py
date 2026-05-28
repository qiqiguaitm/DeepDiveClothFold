"""DAgger collection stack — composition over duplication.

Architecture: includes autonomy_launch.py via IncludeLaunchDescription (gets the
entire policy + slave + camera stack identical to start_autonomy_from_ckpt.sh,
minus rerun — see enable_rerun override below), then adds dagger-only nodes:
  - arm_master_servo × 2 (master arms in CAN-driven servo mode; toggle between
    subscribe (mirror policy) and publish (encoder publish, drag mode))
  - dagger_recorder (state machine + episode recording to Task_X/dagger/<date-v2>/)
  - dagger_pedal_node (USB pedal evdev listener → /dagger/pedal_toggled,
    each press flips PRE_RECORD ↔ HUMAN_RECORD in dagger_recorder; mirrors
    KAI0 official Space-bar semantics from agilex_openpi_dagger_collect.py)

autonomy_recorder is suppressed via record_enable:=false (dagger has its own
recorder writing the same on-disk format under a different subset).
Rerun viz is suppressed via enable_rerun:=false (dagger session is focused —
GUI viz competes for GPU + adds latency; use playback_launch later to inspect).

Toggle: /dagger/takeover (Bool) — True=master→slave teleop (drag), False=slave→master mirror.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _find_project_root():
    candidate = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if os.path.isdir(os.path.join(candidate, 'kai0')):
        return candidate
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, 'kai0')):
            return d
        d = os.path.dirname(d)
    return os.path.expanduser('~/workspace/deepdive_kai0')


_PROJECT_ROOT = _find_project_root()
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, 'config')

# Master CAN names — leader/follower arms wired separately from slaves
import yaml
def _load_yaml(p):
    if os.path.isfile(p):
        with open(p) as f:
            return yaml.safe_load(f) or {}
    return {}
_calib = _load_yaml(os.path.join(_CONFIG_DIR, 'calibration.yml'))
_hw = _calib.get('hardware', {})
_LEFT_MASTER_CAN  = _hw.get('left_master_can',  'can_left_mas')
_RIGHT_MASTER_CAN = _hw.get('right_master_can', 'can_right_mas')


def generate_launch_description():
    # ── Dagger-specific args (passed through to nodes below) ──
    record_task_arg = DeclareLaunchArgument(
        'record_task', default_value='',
        description='Task name (Task_A/B/...); empty = infer from checkpoint_dir')
    record_prompt_arg = DeclareLaunchArgument(
        'record_prompt', default_value='',
        description='Prompt for tasks.jsonl; empty = read checkpoint train_config.json')
    record_subset_arg = DeclareLaunchArgument(
        'record_subset', default_value='dagger',
        description='Dataset subset (default "dagger" — matches kai0_dagger upstream)')

    # ── Compose autonomy_launch.py: same policy + slave + cameras (no rerun) ──
    # Three key overrides:
    #   execute_mode := true   (policy publishes /master/joint_* immediately)
    #   record_enable := false (autonomy_recorder disabled; dagger has its own)
    #   enable_rerun := false  (rerun viz off; dagger session avoids GPU/latency competition)
    autonomy_launch_path = os.path.join(
        get_package_share_directory('piper'), 'launch', 'autonomy_launch.py'
    )
    autonomy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(autonomy_launch_path),
        launch_arguments={
            'execute_mode': 'true',
            'record_enable': 'false',
            'enable_rerun': 'false',
            # Defer policy_inference_node spawning to the dagger_manager web
            # UI — operator clicks "Start session" after picking a ckpt, which
            # forks start_dagger_session.sh. Avoids loading JAX (~22s) at
            # infra bring-up before the user has even opened the dashboard.
            'enable_policy': 'false',
        }.items(),
    )

    # ── DAgger-only: 2× master_servo (subscribe-driven master arms) ──
    # Boot in 'control' (subscribe) state: master mirrors slave by JointCtrl'ing
    # to /master/joint_left,right (same topic policy publishes to → slave executes).
    # /master/enable toggles to 'publish' state for user-drag teleop.
    master_left = Node(
        package='piper', executable='arm_master_servo_node.py',
        name='piper_master_left', output='screen',
        parameters=[{
            'can_port': _LEFT_MASTER_CAN,
            'speed_percent': 30,
            'publish_rate_hz': 30.0,
            'start_state': 'control',
        }],
        remappings=[
            ('/master/joint_states',           '/master/joint_left'),
            ('/puppet/joint_states',           '/puppet_master/joint_left'),
            ('/master/enable',                 '/teach/master_enable_left'),
            ('/master/linkage_config',         '/teach/master_config_left'),
            ('/master/teach_mode',             '/teach/teach_mode_left'),
            ('/master_controled/joint_states', '/master_controled/joint_left'),
            ('/master/button_pressed',         '/master_button_left'),
        ],
    )
    master_right = Node(
        package='piper', executable='arm_master_servo_node.py',
        name='piper_master_right', output='screen',
        parameters=[{
            'can_port': _RIGHT_MASTER_CAN,
            'speed_percent': 30,
            'publish_rate_hz': 30.0,
            'start_state': 'control',
        }],
        remappings=[
            ('/master/joint_states',           '/master/joint_right'),
            ('/puppet/joint_states',           '/puppet_master/joint_right'),
            ('/master/enable',                 '/teach/master_enable_right'),
            ('/master/linkage_config',         '/teach/master_config_right'),
            ('/master/teach_mode',             '/teach/teach_mode_right'),
            ('/master_controled/joint_states', '/master_controled/joint_right'),
            ('/master/button_pressed',         '/master_button_right'),
        ],
    )

    # ── DAgger-only: state machine + episode recorder ──
    dagger_node = Node(
        package='piper', executable='dagger_recorder_node.py',
        name='dagger_recorder', output='screen',
        parameters=[{
            'task_name':      LaunchConfiguration('record_task'),
            'prompt':         LaunchConfiguration('record_prompt'),
            'subset':         LaunchConfiguration('record_subset'),
            'checkpoint_dir': LaunchConfiguration('checkpoint_dir'),
            'operator':       'dagger',
        }],
    )

    # ── DAgger-only: USB pedal → /dagger/pedal_toggled ──
    # Sibling of web/data_manager/backend/tools/pedal_listener.py (same VID:PID
    # + KEY_F3 + DEBOUNCE_MS defaults, env vars compat). Set SKIP_PEDAL=1 to
    # opt out (node exits cleanly; state machine still works via switches alone,
    # but you lose pedal-gated PRE_RECORD → HUMAN_RECORD toggling).
    pedal_node = Node(
        package='piper', executable='dagger_pedal_node.py',
        name='dagger_pedal', output='screen',
    )

    # Stagger so master_servo doesn't init before CAN/cameras are stable
    master_left_delayed  = TimerAction(period=8.0,  actions=[master_left])
    master_right_delayed = TimerAction(period=8.5,  actions=[master_right])
    dagger_delayed       = TimerAction(period=25.0, actions=[dagger_node])
    # Pedal can come up immediately — it doesn't need CAN/cameras/policy.
    pedal_delayed        = TimerAction(period=2.0,  actions=[pedal_node])

    return LaunchDescription([
        record_task_arg, record_prompt_arg, record_subset_arg,
        autonomy,  # includes mode_arg/gpu_arg/config_arg/ckpt_arg/etc. transitively
        master_left_delayed,
        master_right_delayed,
        dagger_delayed,
        pedal_delayed,
    ])
