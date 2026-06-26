#!/usr/bin/env bash
# Submit a 2-node (2x8 A100) ablation job (t1a/t1b/t1c) on the serverless cluster.
# 2 nodes (vs 5) → serverless reschedules far faster after a preemption (mitigates the
# ~70min latency). faultTolerance + retry=10 + save every 250 → self-heals through preemptions.
# Usage:  AIHC_IMG_PASSWORD='Vis@2026' TAG=t1c bash submit_ablation_2n8g.sh
set -euo pipefail
: "${AIHC_IMG_PASSWORD:?set AIHC_IMG_PASSWORD}"
: "${TAG:?set TAG=t1b or t1c}"
COS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
export J_TAG="$TAG" J_PW="$AIHC_IMG_PASSWORD"
export J_CKPT="$COS/wam_fold_wm_runs/train_out_${TAG}_2n8g"
export J_CACHE="$COS/wam_fold_wm_runs/latent_cache_${TAG}"
export J_TOML="$COS/wam_fold_wm/train/recipe_wm_nano_${TAG}.toml"
export J_STEPS="${MAX_STEPS:-4000}"

/mnt/pfs/p46h4f/cosmos/.venv/bin/python3 - << 'PY'
import os, configparser
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.aihc.aihc_client import AihcClient
from baidubce.services.aihc.modules.job.job_model import *

tag = os.environ['J_TAG']; pw = os.environ['J_PW']; steps = os.environ['J_STEPS']
c = configparser.ConfigParser(); c.read('/root/.aihc/config')
config = BceClientConfiguration(
    credentials=BceCredentials(c['default']['access_id'], c['default']['access_key']),
    endpoint='aihc.bj.baidubce.com')
client = AihcClient(config)

IMAGE = 'ccr-249evs6f-vpc.cnc.bj.baidubce.com/visrobot/cosmos:v5.0_QHcnix_20260531063553_QHcnix_20260605060355'
CMD = 'bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/train/aihc/run_train_aihc_cosmos_wm.sh'

job_spec = JobSpec(
    image=IMAGE, replicas=2,
    imageConfig=ImageConfig(username='root', password=pw),
    resources=[Resource('baidu.com/a100_80g_cgpu', 8), Resource('rdma/hca', 1), Resource('sharedMemory', 0)],
    envs=[Env('NUM_GPUS', '8'), Env('NNODES', '2'), Env('REPLICATE_DEGREE', '2'),
          Env('MAX_STEPS', steps), Env('SAVE_ITER', '250'), Env('SCHED_CYCLE', steps),
          Env('CKPT_DIR', os.environ['J_CKPT']),
          Env('WAM_WM_LATENT_CACHE', os.environ['J_CACHE']),
          Env('TOML', os.environ['J_TOML']),
          Env('CUDA_DEVICE_MAX_CONNECTIONS', '1'), Env('NCCL_DEBUG', 'WARN'),
          Env('NCCL_IB_DISABLE', '0'), Env('LOG_COLLECTION', 'true')],
    enableRDMA=True, hostNetwork=True,
)
ds_pfs = Datasource(type='pfsl2', name='pfs-fDgaop', mountPath='/mnt/pfs/p46h4f', sourcePath='/visdata',
    options={'sizeLimit': 0, 'medium': '', 'readOnly': False, 'pfsL1ClusterPort': '8888',
             'pfsL2MountTargetId': ['mt-zSSaab'], 'pfsL2HostMountPath': '/pfs/visdata',
             'cfsInstanceId': '', 'cfsMountPoint': ''})

resp = client.CreateJob(
    resourcePoolId='aihc-serverless', queueID='aihcq-z4v1apdppzwy',
    name=f'cosmos-acwm-{tag}-2n8g',
    command=CMD, jobSpec=job_spec, dataSources=[ds_pfs],
    faultTolerance=True, faultToleranceArgs='--max-num-of-unconditional-retry=3',
    priority='high',
)
print(f"[submit] {tag}: Job {resp.jobName}/{resp.jobId} created (2n8g, save250, retry3)")
PY
