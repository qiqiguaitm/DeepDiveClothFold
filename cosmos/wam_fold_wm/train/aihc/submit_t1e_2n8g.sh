#!/usr/bin/env bash
# Submit t1e (L3 SPATIAL: EVAC 3-cam stacked action map channel-concat) on the serverless cluster
# (2x8 A100). Uses both local (t1d) + cluster (t1e) per the mandate. The spatial cond_tokens are
# built in the packer from the precomputed maps under WAM_ACTMAP_CACHE (shared PFS); code changes
# live on PFS so the container picks them up. Recipe = t1a (L1 regime) + spatial conditioning.
# Usage:  AIHC_IMG_PASSWORD='Vis@2026' bash submit_t1e_2n8g.sh
set -euo pipefail
: "${AIHC_IMG_PASSWORD:?set AIHC_IMG_PASSWORD}"
COS=/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos
export J_PW="$AIHC_IMG_PASSWORD"
export J_CKPT="$COS/wam_fold_wm_runs/train_out_t1e_2n8g"
export J_CACHE="$COS/wam_fold_wm_runs/latent_cache_t1a"   # video latents (unchanged; cond maps are separate)
export J_TOML="$COS/wam_fold_wm/train/recipe_wm_nano_t1a.toml"
export J_ACTMAP="$COS/wam_fold_wm_runs/actmap_cache_3cam"
export J_STEPS="${MAX_STEPS:-4000}"

/mnt/pfs/p46h4f/cosmos/.venv/bin/python3 - << 'PY'
import os, configparser
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.aihc.aihc_client import AihcClient
from baidubce.services.aihc.modules.job.job_model import *

pw = os.environ['J_PW']; steps = os.environ['J_STEPS']
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
          # --- L3 SPATIAL conditioning (t1e) ---
          Env('WAM_COND_SPATIAL', '1'), Env('WAM_COND_CONCAT', '3'),
          Env('WAM_ACTMAP_CACHE', os.environ['J_ACTMAP']),
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
    name='cosmos-acwm-t1e-2n8g',
    command=CMD, jobSpec=job_spec, dataSources=[ds_pfs],
    faultTolerance=True, faultToleranceArgs='--max-num-of-unconditional-retry=3',
    priority='high',
)
print(f"[submit] t1e SPATIAL: Job {resp.jobName}/{resp.jobId} created (2n8g, COND_SPATIAL=1 CONCAT=3, save250)")
PY
