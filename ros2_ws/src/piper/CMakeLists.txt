cmake_minimum_required(VERSION 3.8)
project(piper)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/arm_reader_node.py
  scripts/arm_teleop_node.py
  scripts/master_handle_node.py
  scripts/policy_inference_node.py
  scripts/replay_node.py
  scripts/rerun_viz_node.py
  scripts/multi_camera_node.py
  scripts/video_publisher_node.py
  scripts/joint_state_loopback_node.py
  scripts/autonomy_recorder_node.py
  scripts/dagger_recorder_node.py
  scripts/dagger_pedal_node.py
  scripts/arm_master_servo_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)
# session_launch.py is installed by the DIRECTORY rule above; explicit
# enumeration above for scripts (PROGRAMS) only.

ament_package()
