"""DAgger session launch — spawn only policy_inference_node.

Paired with start_dagger_session.sh which validates the ckpt sidecar
(train_config.json + override_asset_id assets/norm_stats) and exports
OPENPI_EXTRA_CONFIG before invoking this file.

Intended usage:
    ros2 launch piper session_launch.py \
        checkpoint_dir:=<ckpt_dir> \
        config_name:=<base_config> \
        execute_mode:=true

The dagger infra (CAN/cameras/arms/dagger_recorder/dagger_pedal) is
expected to be already up via start_dagger_collect.sh (which sets
enable_policy:=false inside dagger_launch). This launch file does NOT
re-spawn cameras/arms/dagger nodes — that would conflict on CAN and
duplicate the state machine.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _find_kai0_root():
    candidate = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if os.path.isdir(os.path.join(candidate, 'kai0')):
        return os.path.join(candidate, 'kai0')
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, 'kai0')):
            return os.path.join(d, 'kai0')
        d = os.path.dirname(d)
    return '/data1/tim/workspace/deepdive_kai0/kai0'


_KAI0_ROOT = _find_kai0_root()


def generate_launch_description():
    mode_arg = DeclareLaunchArgument('mode', default_value='ros2',
        description='ros2 (v0/JAX in-process) | websocket (v1 Triton serve client) | both')
    gpu_arg = DeclareLaunchArgument('gpu_id', default_value='0',
        description='GPU ordinal for JAX (CUDA_VISIBLE_DEVICES)')
    config_arg = DeclareLaunchArgument('config_name',
        default_value='pi05_flatten_fold_normal_v1',
        description='openpi base_config_name (from sidecar train_config.json)')
    ckpt_arg = DeclareLaunchArgument('checkpoint_dir',
        default_value='',
        description='Packed ckpt dir (must contain train_config.json + assets/<asset_id>/norm_stats.json)')
    prompt_arg = DeclareLaunchArgument('prompt',
        default_value='Flatten and fold the cloth.',
        description='Language prompt')
    execute_mode_arg = DeclareLaunchArgument('execute_mode', default_value='true',
        description='Start executing (publish /master/joint_*) immediately')
    enable_rtc_arg = DeclareLaunchArgument('enable_rtc', default_value='true',
        description='Pi0RTCConfig with guidance/horizon')
    # ── WebSocket client (v1 Triton path) ──────────────────────────────
    # mode:=websocket makes policy_inference_node a thin client to a V1
    # serve_policy_v1.py server (started by start_dagger_session.sh on
    # --serve-port). config_name/checkpoint_dir are ignored in this mode —
    # the server owns the weights.
    host_arg = DeclareLaunchArgument('host', default_value='localhost',
        description='V1 serve host (mode=websocket only)')
    port_arg = DeclareLaunchArgument('port', default_value='8000',
        description='V1 serve port (mode=websocket only; start_dagger_session.sh v1 uses 8002)')
    # ── RTC / smoothing / obs knobs (defaults = JAX legacy sizing, matching
    #    autonomy_launch.py + policy_inference_node.py declare_parameter).
    #    start_dagger_session.sh v1 branch overrides these to the validated
    #    V1 production config (20Hz / k=6 / exec_h=12 / shm). ────────────
    inference_rate_arg = DeclareLaunchArgument('inference_rate', default_value='3.0',
        description='Hz of inference loop (JAX legacy 3.0, V1 overrides 20.0)')
    latency_k_arg = DeclareLaunchArgument('latency_k', default_value='8',
        description='Head-trim steps of new chunk (JAX legacy 8, V1 overrides 6)')
    min_smooth_steps_arg = DeclareLaunchArgument('min_smooth_steps', default_value='8',
        description='Min blend window for chunk overlap smoothing (JAX 8, V1 8)')
    rtc_execute_horizon_arg = DeclareLaunchArgument('rtc_execute_horizon', default_value='16',
        description='RTC guidance horizon (JAX legacy 16, V1 overrides 12)')
    publish_rate_arg = DeclareLaunchArgument('publish_rate', default_value='30',
        description='Hz of command publish loop (JAX legacy 30, V1 overrides 80)')
    publish_smooth_alpha_arg = DeclareLaunchArgument('publish_smooth_alpha', default_value='0.5',
        description='Publish-time EMA factor (Layer 1.1E); shared across v0/v1')
    rtc_smooth_method_arg = DeclareLaunchArgument('rtc_smooth_method', default_value='min_jerk',
        description='RTC chunk-overlap weight curve: min_jerk | linear')
    fast_obs_pipeline_arg = DeclareLaunchArgument('fast_obs_pipeline', default_value='false',
        description='V1: bypass JPEG mapping + CvBridge + BGR<->RGB. JAX keeps false.')
    pipelined_obs_arg = DeclareLaunchArgument('pipelined_obs', default_value='false',
        description='V1: ObsPrefetchWorker pre-fetch obs. JAX keeps false.')
    transport_arg = DeclareLaunchArgument('transport', default_value='ws',
        description='V1: ws (default) | shm (POSIX shm, low-latency). JAX path n/a.')

    # Mirror env variables data_manager uses for venv resolution. Path order:
    # kai0/.venv site-packages → .pth-derived dirs → kai0/src. Keep identical
    # to start_autonomy.sh's setup so V1 imports + JAX work.
    import glob
    venv_lib = os.path.join(_KAI0_ROOT, '.venv', 'lib')
    pydirs = sorted(glob.glob(os.path.join(venv_lib, 'python3.*')))
    venv_sp = os.path.join(pydirs[-1], 'site-packages') if pydirs else os.path.join(venv_lib, 'python3.12', 'site-packages')
    pth_dirs = []
    if os.path.isdir(venv_sp):
        for pth in sorted(os.listdir(venv_sp)):
            if pth.endswith('.pth'):
                try:
                    with open(os.path.join(venv_sp, pth)) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and os.path.isabs(line):
                                pth_dirs.append(line)
                except Exception:
                    pass
    pythonpath = ':'.join([venv_sp] + pth_dirs + [os.path.join(_KAI0_ROOT, 'src')])

    set_py = SetEnvironmentVariable(
        'PYTHONPATH',
        pythonpath + ':' + (os.environ.get('PYTHONPATH') or ''),
    )
    # JAX compilation cache lives in the project dir, persists across sessions.
    set_cache = SetEnvironmentVariable('JAX_COMPILATION_CACHE_DIR',
        os.path.join(_KAI0_ROOT, '.xla_cache'))
    # GPU memory: grow-on-demand, NOT upfront-grab. The old config
    # (PREALLOCATE unset → default true, MEM_FRACTION=0.9) made JAX reserve
    # 0.9×32GB ≈ 28.8GB at init. On sim01 every GPU is shared with other jobs,
    # so no card has 28.8GB free → the preallocation blocks indefinitely and
    # create_trained_policy hangs forever right after the Pi0→Pi0RTC upgrade
    # (observed 2026-06-05: web "Start session" never completes). pi05 inference
    # needs ~10-13GB, so cap growth at 0.35 (≈11GB) and let it allocate lazily —
    # matching the working autonomy_launch.py (PREALLOCATE=false, MEM_FRACTION=0.35).
    set_prealloc = SetEnvironmentVariable('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    set_mem_frac = SetEnvironmentVariable('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.35')

    policy_node = Node(
        package='piper', executable='policy_inference_node.py',
        name='policy_inference', output='screen',
        parameters=[{
            'mode': LaunchConfiguration('mode'),
            'config_name': LaunchConfiguration('config_name'),
            'checkpoint_dir': LaunchConfiguration('checkpoint_dir'),
            'prompt': LaunchConfiguration('prompt'),
            'gpu_id': LaunchConfiguration('gpu_id'),
            'execute_mode': LaunchConfiguration('execute_mode'),
            'enable_rtc': LaunchConfiguration('enable_rtc'),
            # WebSocket client (v1) — server host/port. Ignored in mode=ros2.
            'host': LaunchConfiguration('host'),
            'port': LaunchConfiguration('port'),
            # RTC / smoothing / obs knobs (typed to match autonomy_launch.py).
            'inference_rate': LaunchConfiguration('inference_rate'),
            'latency_k': LaunchConfiguration('latency_k'),
            'min_smooth_steps': LaunchConfiguration('min_smooth_steps'),
            'rtc_execute_horizon': LaunchConfiguration('rtc_execute_horizon'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'publish_smooth_alpha': LaunchConfiguration('publish_smooth_alpha'),
            'rtc_smooth_method': LaunchConfiguration('rtc_smooth_method'),
            'fast_obs_pipeline': ParameterValue(LaunchConfiguration('fast_obs_pipeline'), value_type=bool),
            'pipelined_obs': ParameterValue(LaunchConfiguration('pipelined_obs'), value_type=bool),
            'transport': ParameterValue(LaunchConfiguration('transport'), value_type=str),
            # Required topics (match autonomy_launch defaults). The infra
            # stack has these topics already publishing, so we just connect.
            'img_front_topic': '/camera_f/camera/color/image_raw',
            'img_left_topic':  '/camera_l/camera/color/image_raw',
            'img_right_topic': '/camera_r/camera/color/image_raw',
            'puppet_left_topic':  '/puppet/joint_left',
            'puppet_right_topic': '/puppet/joint_right',
        }],
    )

    return LaunchDescription([
        set_py, set_cache, set_prealloc, set_mem_frac,
        mode_arg, gpu_arg, config_arg, ckpt_arg, prompt_arg,
        execute_mode_arg, enable_rtc_arg,
        host_arg, port_arg,
        inference_rate_arg, latency_k_arg, min_smooth_steps_arg,
        rtc_execute_horizon_arg, publish_rate_arg, publish_smooth_alpha_arg,
        rtc_smooth_method_arg, fast_obs_pipeline_arg, pipelined_obs_arg,
        transport_arg,
        policy_node,
    ])
