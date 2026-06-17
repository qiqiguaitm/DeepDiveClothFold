# Action 表示与处理 — 四系统横向对比

> 生成时间：2026-06-15  
> 覆盖系统：kai0 / gwp-head（官方 tag）/ gwp 本地 HEAD / fastwam v4

---

## 0. 背景与问题定义

gwp-ori v3_abs 训练（50k steps, 5×8×A100）出现 MAE@48=0.44-0.46 平台——action_loss 已收敛到 1 但 eval 不动。排查后确认是 **代码回归 + 真 abs 在 flow_shift=5 下低噪信号不足**，而非超参问题。本文对比四个系统在 action 处理上的设计差异，作为诊断记录和后续决策依据。

---

## 1. 六维度对比

### 1.1 动作表示（Delta vs Abs）

| 系统 | 表示方式 | 哪些维度做 delta |
|------|---------|----------------|
| **kai0** | piper14 mixed | joints 0-5, 7-12 → delta；grippers 6, 13 → abs |
| **gwp 官方 tag** | piper14 mixed（同 kai0） | 同上，训练/推理均硬编码 |
| **gwp 本地 HEAD** | 从 stats JSON 读取 | v3_abs：`delta_mask=[F]×14` → 真 abs；旧 stats 无字段时回退 piper14 |
| **fastwam v4** | 真 abs | `action_state_transforms: null`，代码里完全无 delta 分支 |

**关键发现**：gwp 官方 tag 因 `_piper14` 硬编码，实际运行的是 delta（即便 stats 是 abs 统计），造成归一化偏置（normalized_delta ≈ -1.9，近常数）。模型学会"预测当前 state"，MAE 虚低。这是 gwp v3_abs 早期"表现良好"的真实原因，本质是 bug，非设计优势。

---

### 1.2 Mask / 表示方式的来源

```
kai0:
  config.py 显式 flag
    use_delta_joint_actions: bool = True
    delta_action_mask = make_bool_mask(6, -1, 6, -1)   # piper14

gwp 官方 tag:
  transforms.py 硬编码（五处散落）
    _piper14 = [T,T,T,T,T,T,F,T,T,T,T,T,T,F]
    delta_mask_templates = {0: _piper14, 1: _piper14}
  inference_server.py 命令行默认值（单独一份）
    --delta_mask "1,1,1,1,1,1,0,1,1,1,1,1,1,0"
  ← 训练 mask 和推理 mask 源头分离，存在漂移风险

gwp 本地 HEAD:
  stats JSON 内嵌 delta_mask（单一真值源）
    resolve_delta_mask(stats_dict, d, fallback=piper14)
  训练/推理/eval 均从同一 JSON 读取
  ← 换 stats 文件不影响代码；老 stats 无字段时静默回退 piper14（风险点）

fastwam v4:
  config.yaml 显式 null
    action_state_transforms: null
  代码里完全无 delta 分支，换 stats 不影响表示方式
  ← 最强保证，无回退路径
```

---

### 1.3 归一化

| 系统 | 归一化对象 | mean/std 量级（action joints） |
|------|-----------|-------------------------------|
| kai0 | z-score(delta_action) | mean≈0，std≈0.1-0.35 |
| gwp 官方 tag | z-score(delta_action) | 同上（piper14 delta 统计） |
| gwp 本地 HEAD v3_abs | z-score(abs_action) | mean≈±1.4 rad（绝对关节角），std≈0.3-0.6 |
| fastwam v4 | z-score(abs_action)，`norm_default_mode: z-score` | 同上 |

判断一份 stats 是 delta 还是 abs 的最快依据：action joints 的 q01/q99 范围：
- delta：±0.5 ~ 1.2 rad（小）
- abs：±2.3 rad（与 state 分布相同，大）

---

### 1.4 Training Weight（损失加权）

flow_shift=5 下，sigma 采样分布：$\sigma = \frac{5u}{1+4u}$，u~Uniform(0,1)。约 **88% 样本落在 σ>0.4（高噪区）**，低噪区（σ<0.2）训练信号极弱。对真 abs action 影响尤为显著——高噪下模型只需预测均值，无法学到精细的关节角轨迹。

```
kai0:     无 training_weight（delta 天然收敛，不需要）
gwp tag:  无 training_weight
gwp HEAD: training_weight=True（2026-06-15 新增）
fastwam:  training_weight（对 video 和 action 均加权）

公式（fastwam scheduler_continuous.py，gwp HEAD 完全对齐）：
  预计算（基于 flow_shift-warped 分布）：
    u_grid = linspace(1,0,1001)[:-1]
    sigma_grid = flow_shift·u / (1+(flow_shift-1)·u)
    t_grid = sigma_grid × 1000
    y_grid = exp(-2·((t_grid-500)/1000)²)
    y_min = min(y_grid)
    norm_const = mean(y_grid - y_min)

  每步：
    w(t) = (exp(-2·((t-500)/1000)²) - y_min) / norm_const
    loss = (per_sample_loss × w(t)).mean()

效果：t=500(σ=0.5)处权重最高≈2.0，t=900(σ=0.9)处权重≈0.2，
把有效训练质量从高噪区拉回低噪区。
```

---

### 1.5 Action Sigma（与 Video 共享 or 独立）

```
kai0:           N/A（JAX π0.5，非 flow-matching video joint 训练）

gwp tag:        共享
                action_sigma = sigma.squeeze(-1).squeeze(-1)
                # sigma 是视频的采样，action 用同一个值

gwp HEAD:       默认共享；可选 async_noise（ANS，t_video ≥ t_action 耦合）
                新增 independent_action_sigma=True（2026-06-15）
                if independent_action_sigma:
                    ans_action_ts, ans_action_sigma = get_timestep_and_sigma(bs, ndim=3)
                    # 完全独立重采样，无耦合约束

fastwam v4:     完全独立
                timestep_action = train_action_scheduler.sample_training_t(...)
                # train_action_scheduler 是独立的 ContinuousFlowMatchingScheduler 实例
                # timestep_video 和 timestep_action 互不相关
```

**ANS vs 完全独立的区别**：ANS 强制 t_video ≥ t_action（视频噪声恒高于动作噪声），适配"推理时先去噪动作再去噪视频"的联合解码场景。fastwam v4 无此约束，两者完全解耦，这是 gwp HEAD 的 `independent_action_sigma` 对应的语义。

---

### 1.6 推理侧：Delta → Abs 重建

```
kai0:
  model_transforms 输出 apply AbsoluteActions(mask)
  actions[mask] += state[mask]
  # 在 openpi transform pipeline 的 output 侧自动反算

gwp 官方 tag:
  add_state_to_action(action, state, mask=delta_mask)
  delta_mask 来自 inference_server.py 命令行参数 --delta_mask
  ← 与训练 mask 来源分离（不同代码路径，需人工保持一致）

gwp 本地 HEAD:
  add_state_to_action(action, state, mask=resolve_delta_mask(stats))
  从同一份 stats JSON 读取（单一真值源）
  v3_abs：delta_mask=[F]×14 → add_state_to_action 是 no-op（纯 denorm）

fastwam v4:
  无需重建（全 abs，直接 denormalize 即是目标关节角）
  post-processing: denorm → add_state_to_action(null mask → no-op)
```

---

## 2. 架构差异（影响 delta/abs 选择的根因）

| 系统 | Action 解码架构 | 对 delta/abs 的影响 |
|------|---------------|-------------------|
| kai0（π0.5） | 独立 action expert（diffusion head，不共享视频 token） | abs/delta 对 backbone 无关 |
| gwp（CasualWATransformer） | action token 与 video token **共享同一 transformer**，causal mask 区分 | action token 需与 video token "同语义空间"；video flow-matching 目标是 velocity（变化量），delta action 天然对齐 |
| fastwam v4 | **独立 ActionDiT**（378M，MoE expert，与 Wan backbone 分离） | ActionDiT 有自己的 embedding 空间，abs 不需要与 video 对齐 |

**结论**：gwp 共享 transformer 的架构天然偏好 delta（变化量与 video velocity 语义对齐）；fastwam v4 的独立 ActionDiT 则无此约束，abs 更合理。这是两者在 delta/abs 选择上分歧的架构根因。

---

## 3. 汇总对比表

| 维度 | kai0 | gwp 官方 tag | gwp 本地 HEAD | fastwam v4 |
|------|------|------------|--------------|-----------|
| 动作表示 | piper14 delta | piper14 delta | stats JSON（v3: 真 abs） | 真 abs |
| Mask 来源 | config 显式 flag ✓ | 代码硬编码 × | stats JSON（软约束） △ | config null（最强） ✓ |
| 训练/推理 mask 一致性 | ✓（同一 flag） | ✗（两处分离） | ✓（同一 stats） | ✓（配置 null） |
| 归一化 | z-score(delta) | z-score(delta) | z-score(abs) | z-score(abs) |
| training_weight | ✗ | ✗ | ✓（2026-06-15 新增） | ✓ |
| Video loss 加权 | — | ✗ | ✓ | ✓ |
| Action sigma 独立 | N/A | ✗（共享） | ✓（2026-06-15 新增） | ✓（独立 scheduler） |
| 推理重建单一真值源 | ✓ | ✗ | ✓ | ✓（无需重建） |
| Action 解码架构 | 独立 expert | 共享 transformer | 共享 transformer | 独立 ActionDiT（378M） |

---

## 4. 为什么 v3_abs 训练失败

| 环节 | 问题 |
|------|------|
| 代码回归 | gwp HEAD 把 `_piper14` 硬编码改成 `resolve_delta_mask(stats)`；v3_abs 的 stats `delta_mask=[F]×14` → 首次真正走 abs 路径；之前"正常"是 bug 掩盖 |
| flow_shift=5 | 83% 样本在 σ>0.5 高噪区，低噪区梯度极弱 |
| 共享 transformer | action token 与 video token 同空间，abs 绝对关节角与 video velocity 语义不对齐 |
| 无 training_weight | 无法补偿高噪区采样偏置 |

**结果**：模型收敛到"预测 action 均值"（pred_std ≈ 0.08 vs gt_std ≈ 0.15-0.95），MAE@48 平台在 0.44-0.46。

---

## 5. gwp_abs_v4 新增的对齐（2026-06-15）

在 `wa_casual_trainer.py` + `visrobot01_gwp_abs_v4.py` 中新增，对齐 fastwam v4：

```python
# 1. training_weight（wa_casual_trainer.py get_models）
self.training_weight_enabled = bool(model_config.get("training_weight", False))
# 预计算 _tw_y_min, _tw_norm_const（基于 flow_shift warped 分布）

# 2. 应用到 loss（forward_step）
action_loss_per = action_loss.mean(dim=(1, 2))       # [bs]
tw = self._training_weight(action_sigma.reshape(bs) * 1000)
action_loss = (action_loss_per * tw).mean()

visual_loss_per = visual_loss.mean(dim=range(1, ndim)) # [bs]
visual_loss = (visual_loss_per * self._training_weight(_ts_per_sample)).mean()

# 3. independent_action_sigma（forward_step）
if self.independent_action_sigma and not self.async_noise:
    ans_action_ts, ans_action_sigma = self.get_timestep_and_sigma(_bs, ndim=3)
    # → action_sigma 完全独立于 video sigma

# 4. action token 填入独立 timestep（填入 transformer 的 timestep 向量）
if self.async_noise or self.independent_action_sigma:
    a0 = num_state_tokens + num_clean_latent_tokens
    timestep[:, a0:a0 + num_action_tokens] = ans_action_ts[:, None]
```

**config 开关**（`visrobot01_gwp_abs_v4.py`，继承 v3_abs 全部设置，仅改 project_dir）：
```python
config["models"]["training_weight"] = True
config["models"]["independent_action_sigma"] = True
config["project_dir"] = "runs/gwp_abs_v4"
```

**AIHC job**：job-j9lzd7f7euaj（gwp-abs-v4，5×8×A100，50k steps）

---

## 6. 决策参考

### 优先 delta（若 gwp_abs_v4 结果不显著优于 gwp_ori）

| 条件 | 理由 |
|------|------|
| 共享 transformer 架构 | delta（变化量）与 video velocity 语义对齐，无需额外 trick |
| 收敛速度 | 实测 1k 步 delta MAE@48=0.235 vs abs=0.303（1.29×优） |
| 工程安全 | 不依赖 training_weight/independent_sigma 等补丁，故障面更小 |
| 参考：gwp_ori | 实质 piper14 delta，MAE@48=0.0916，无额外补丁 |

### 优先 abs（若 gwp_abs_v4 收敛且 MAE@48 < 0.088）

| 条件 | 理由 |
|------|------|
| 长 horizon 一致性 | 48-step chunk 中 abs 无 delta 不同步问题 |
| 跨 embodiment 扩展 | abs 归一化在不同 state 分布下更鲁棒 |
| fastwam 路线对齐 | 未来若切独立 ActionDiT，abs 是正确底座 |

### 判决阈值

- gwp_abs_v4 MAE@48 < **0.088**（优于 gwp_ori 0.0916 超过 3 pts）：保留 abs，training_weight 值得
- gwp_abs_v4 MAE@48 ≥ 0.088：回到 piper14 delta，去掉 training_weight/independent_sigma，对齐 kai0

---

## 7. 复现信息

| 系统 | 关键文件 |
|------|---------|
| kai0 delta | `kai0/src/openpi/training/config.py` → `LerobotAgilexDataConfig.use_delta_joint_actions` |
| kai0 transforms | `kai0/src/openpi/transforms.py` → `DeltaActions` / `AbsoluteActions` |
| gwp 官方 tag（历史） | `wa_transforms_lerobot.py` 的 `_piper14` + `delta_mask_templates`（已被 `resolve_delta_mask` 替换） |
| gwp HEAD mask 解析 | `world_action_model/pipeline/utils.py` → `resolve_delta_mask()` |
| gwp HEAD trainer | `world_action_model/trainer/wa_casual_trainer.py` → `training_weight_enabled`, `independent_action_sigma`, `_training_weight()` |
| gwp abs v4 config | `world_action_model/configs/visrobot01_gwp_abs_v4.py` |
| fastwam v4 scheduler | `fastwam/src/fastwam/models/wan22/schedulers/scheduler_continuous.py` → `training_weight()` |
| fastwam v4 forward | `fastwam/src/fastwam/models/wan22/fastwam.py` → `train_action_scheduler.sample_training_t()` |
| fastwam v4 config | `fastwam/runs/visrobot01_fold_uncond_1e-4/aihc_5n8g_v4/config.yaml` |
