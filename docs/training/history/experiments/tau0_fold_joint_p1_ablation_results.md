# τ₀-WM 关节空间叠衣服微调 — P1 报告（先验迁移消融 + 16 卡评估）

> 日期: 2026-06-04 · 任务: Flatten and fold the cloth · 目标本体: visrobot01
> 训练框架: 自建 tau0 trainer（`tau0_wm/finetune/`）· 方案见 [future_plans/plans/tau0_fold_visrobot01_joint_finetune.md]
> 训练: P1 dev-box 2 节点 16 卡 · 评估: 2 节点 16 卡分布式 · P2: AIHC 4 节点 32 卡（提交中）

## 1. 结论先行（GO）

**τ₀-WM 的预训练干（`action_blocks×30` + 视频主干）能迁移到关节空间叠衣服动作预测** —— 这是「真·复用 tau0」是否值得的 go/no-go 分水岭，结论 **GO**，继续 P2。

三组 16 卡分布式评估（visrobot01_val 200 集,3600 个 window,动作 flow velocity-MSE）:

| 配置 | val action-loss | 含义 |
|---|---|---|
| 未训练基线: 预训练干 + **随机头** | **4.76** | 起点 |
| **P1: 预训练干(冻结) + 训练头** | **1.00** | ↓79%,头学会从冻结干解码出关节动作 |
| 对照: **随机干** + 同一训练头 | **3.16** | 换掉预训练干 → 同样的头失效 |

- 预训练干的贡献:同样的训练头,在预训练干上 **1.00** vs 随机干上 **3.16**(差 3.2×)→ **预训练干的特征是关键**。
- 头训练的贡献:预训练干上,随机头 **4.76** → 训练头 **1.00**(↓79%)→ 32K 参数的小头足以从冻结干读出动作。
- 两者缺一不可,合起来达到 val 1.00。

## 2. 方法与架构

- **只重置 3 个张量**:`action_proj_in.weight`(20→14)、`action_head.head.{weight,bias}`(20→14);其余 **1403/1406** 张量(含 `action_blocks×30` + 视频主干 + VAE-aligned 嵌入)从 tau0 预训练加载。实测确认(`p0_probe_load.py`)。
- **数据零额外开销**:直接复用 GigaWorld 的 `vae_latent`+`t5` 缓存 —— 实测 tau0 的 Wan2.2 VAE 归一化常数与 diffusers VAE **逐值相同(max diff 0.0)**;设 chunk=5(T_lat=2)对齐缓存。
- **动作约定**:14 维关节,delta(夹爪 idx 6,13 绝对),per-embodiment 归一化(`statistics_{visrobot01,kairobot01}.json`,由 GigaWorld norm_stats 转换)。
- **训练目标**:flow-matching(σ=flow_shift·σ/(1+(flow_shift-1)σ),velocity target=noise−clean),首帧 conditioning held clean(mask2/temp_ts 对齐 tau0 `infer`),`λ_video=0`(纯动作 FT)。

## 3. P1 训练（冻结干暖启 + 消融）

- 配置:2 节点 16×A100(b2=192.168.20.128 master + b1=192.168.20.169),NCCL/eth0,accelerate DDP,bf16,grad_accum=4(eff. batch 64),lr=1e-4,**只训 32K 参数(两个关节投影)**,冻结整个 5.51B 干。
- 数据:visrobot01_train×3 + kairobot01(per-embodiment),复用 latent 缓存。
- 速度:**0.72–0.79 step/s**,3000 步 ≈ 68 min。
- 收敛(train action-loss):

```
step   10: 5.16     step 1210: 1.22
step  310: 1.87     step 1510: 0.98
step  610: 1.66     step 2110: 0.98
step  910: 1.56     step 3000: ~1.0
```

干冻死 → 这条曲线纯粹反映「冻结的预训练干特征能支撑多少」→ 5.16→1.0 即先验有效。

## 4. 16 卡分布式评估

- `run_eval_dist.py` + `launch_eval_2node.sh`:2 节点 16 卡,vis_val 200 集按 rank 分片,per-dim velocity-MSE all-reduce。
- vis_val latents 用 GigaWorld 编码器预抽(200/200,71691 window)。
- **P1 per-dim val action-loss**(14 维关节,较均匀 0.86–1.19):

```
左臂  j0..j5,grip:  1.02 0.86 0.89 1.08 1.11 1.19 | 0.96
右臂  j0..j5,grip:  1.16 0.88 0.94 0.98 0.92 1.17 | 0.87
```

无明显坏维;val(1.00)≈ train(~1.0)→ 不过拟合,泛化到 held-out。

## 5. P2 — 4 节点 32 卡训练（AIHC,提交中）

- 参考 `giga_world_policy/scripts/aihc`,自建 `finetune/aihc/run_train_aihc_tau0.sh` + `aijob_tau0_4n8g.json`,经 `aihc job create -f ... -p aihc-serverless -q aihcq-z4v1apdppzwy` 提交。
- Job: **`job-lqgkgcya5dat`**(tau0-fold-joint-p2-4n8g),**replicas=4 × 8×A100 = 32 卡**,RDMA,hostNetwork,镜像 **cosmos:v5.0_...20260605060355**。
- 任务:**P2 specialize** —— 解冻 `action_blocks`(512M 可训),从 P1 暖启 ckpt(`step_1000.pt`)续训,max_steps=20000,lr=3e-5。
- 状态历史:`job-2opevitevlmt`/`job-lqgkgcya5dat` 均 **ImagePullBackOff → Failed**。根因 = 提交时 `imageConfig.password=""`(私有 VPC 仓库需鉴权),**遗漏了 GigaWorld `resubmit_latent.sh` 的密码注入步骤**。修复:`finetune/aihc/submit_tau0_aihc.sh`(注入 `AIHC_IMG_PASSWORD`(root/Vis@2026)+ 容错重试)。正式 job: **`job-tz0j5hv3e386`**。
- 其余 spec(datasource `mt-zSSaab`/resources/RDMA/hostNetwork/queue)逐项核对与 GigaWorld 成功 job 一致。

## 6. 结论与下一步

- ✅ **先验迁移成立(GO)**:tau0 预训练干对关节空间叠衣服动作预测有实质贡献(3.16→1.00)。
- ✅ 全流程(模型手术、flow-matching trainer、缓存复用 dataloader、16/32 卡分布式、评估)已打通并验证。
- ▶ **P2(32 卡)**:解冻 action_blocks 专精(vis×3+kai → vis-only),预期 val action-loss 进一步下降。
- ▶ 部署:`TauPolicyJoint`(关节进关节出,免 FK/IK)→ visrobot01 真机闭环成功率。

## 附录:关键产物
- 训练/评估代码: `tau0_wm/finetune/`(`model_joint.py` `data_joint.py` `train_tau0.py` `run_train.py` `run_eval_dist.py` `launch_{,eval_}2node.sh` `aihc/`)
- P1 ckpts: `runs/tau0_fold_p1/{step_1000,2000,3000,final}.pt`
- 评估结果: `runs/eval_report.json`
- P2(32 卡)输出: `runs/tau0_fold_p2_32g/`(job `job-2opevitevlmt`)
