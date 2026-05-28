export type DaggerState =
  | "POLICY_RUN"
  | "ALIGNING"
  | "HUMAN_RECORD"
  | "RETURNING";

export interface CameraHealth {
  fps: number;
  target_fps: number;
  dropped: number;
  latency_ms: number;
}

export interface JointState {
  left_joints: number[];
  right_joints: number[];
  left_gripper: number;
  right_gripper: number;
}

export interface DaggerStatus {
  ts: number;
  // Infra = CAN + cameras + arms + dagger_recorder + dagger_pedal (no policy)
  stack_running: boolean;
  stack_pid: number | null;
  stack_log_path: string | null;
  // Session = policy_inference (forked via dagger_manager web after ckpt picked)
  session_running: boolean;
  session_pid: number | null;
  session_log_path: string | null;
  session_started_at: number | null;
  state: DaggerState | null;
  recording: boolean | null;
  button_left: boolean;
  button_right: boolean;
  policy_execute: boolean | null;
  last_pedal_ts: number | null;
  ros_alive: boolean;
  inference_episodes: number;
  dagger_episodes: number;
  ckpt: string | null;
  task: string | null;
  cameras: Record<string, CameraHealth>;
}

export interface CkptEntry {
  path: string;
  name: string;
  group: string;
  variant: "v0" | "v1";
  has_sidecar: boolean;
  has_norm_stats: boolean;
  has_v1_pkl: boolean;
  config_name: string | null;
  task_hint: string | null;
}

export interface EpisodeEntry {
  subset: "dagger" | "inference";
  date: string;
  episode_id: number;
  length: number;
  duration_s: number;
  operator: string;
  prompt: string;
  success: boolean;
  note: string;
  created_at: number | null;
  has_video: boolean;
}
