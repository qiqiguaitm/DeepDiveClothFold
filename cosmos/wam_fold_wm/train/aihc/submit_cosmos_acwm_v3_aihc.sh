#!/usr/bin/env bash
# Submit the Cosmos3-Nano wam_fold_wm FD world-model 5n8g AIHC job.
# Usage:  AIHC_IMG_PASSWORD='Vis@2026' bash submit_cosmos_wm_aihc.sh
# Uses baidubce v2 SDK (action=CreateJob) — the v1 aihc CLI no longer accepts new jobs.
set -euo pipefail
: "${AIHC_IMG_PASSWORD:?set AIHC_IMG_PASSWORD in env}"

/mnt/pfs/p46h4f/cosmos/.venv/bin/python3 - "$AIHC_IMG_PASSWORD" << 'PY'
import sys, configparser
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.aihc.aihc_client import AihcClient
from baidubce.services.aihc.modules.job.job_model import *

pw = sys.argv[1]
c = configparser.ConfigParser(); c.read('/root/.aihc/config')
config = BceClientConfiguration(
    credentials=BceCredentials(c['default']['access_id'], c['default']['access_key']),
    endpoint='aihc.bj.baidubce.com',
)
client = AihcClient(config)

IMAGE = 'ccr-249evs6f-vpc.cnc.bj.baidubce.com/visrobot/cosmos:v5.0_QHcnix_20260531063553_QHcnix_20260605060355'
CMD   = 'bash /mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm/train/aihc/run_train_aihc_cosmos_wm.sh'

job_spec = JobSpec(
    image=IMAGE, replicas=5,
    imageConfig=ImageConfig(username='root', password=pw),
    resources=[Resource('baidu.com/a100_80g_cgpu', 8), Resource('rdma/hca', 1), Resource('sharedMemory', 0)],
    envs=[Env('NUM_GPUS','8'), Env('NNODES','5'), Env('REPLICATE_DEGREE','5'),
          Env('MAX_STEPS','10000'), Env('SAVE_ITER','500'), Env('SCHED_CYCLE','10000'),
          Env('CKPT_DIR','/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/train_out_v3_5n8g'),
          Env('WAM_WM_LATENT_CACHE','/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/wam_fold_wm_runs/latent_cache_v3'),
          Env('CUDA_DEVICE_MAX_CONNECTIONS','1'), Env('NCCL_DEBUG','WARN'),
          Env('NCCL_IB_DISABLE','0'), Env('LOG_COLLECTION','true')],
    enableRDMA=True, hostNetwork=True,
)
ds_pfs = Datasource(type='pfsl2', name='pfs-fDgaop', mountPath='/mnt/pfs/p46h4f',
    sourcePath='/visdata',
    options={'sizeLimit':0,'medium':'','readOnly':False,'pfsL1ClusterPort':'8888',
             'pfsL2MountTargetId':['mt-zSSaab'],'pfsL2HostMountPath':'/pfs/visdata',
             'cfsInstanceId':'','cfsMountPoint':''})

resp = client.CreateJob(
    resourcePoolId='aihc-serverless',
    queueID='aihcq-z4v1apdppzwy',
    name='cosmos-acwm-fold-v3-5n8g',
    command=CMD, jobSpec=job_spec,
    dataSources=[ds_pfs],
    faultTolerance=True,
    faultToleranceArgs='--max-num-of-unconditional-retry=3',
    priority='normal',
)
print(f"[submit] Job {resp.jobName}/{resp.jobId} created successfully")
PY
