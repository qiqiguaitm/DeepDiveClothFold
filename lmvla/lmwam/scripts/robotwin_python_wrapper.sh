#!/bin/bash
export VK_ICD_FILENAMES=/vePFS/HuanQian/conda_envs/RoboTwin/lib/python3.10/site-packages/sapien/vulkan_library/nvidia_icd.json
export PYTHONPATH=/vePFS/tim/robotwin_client_deps:$PYTHONPATH
exec /vePFS/HuanQian/conda_envs/RoboTwin/bin/python "$@"
