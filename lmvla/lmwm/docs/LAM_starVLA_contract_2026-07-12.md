# LAM↔starVLA I/O 契约(P0 产出, 2026-07-12)

> LMWM adapter 要实现的接口。源码在 `lmvla/lawam/`,核心 `starVLA/model/framework/vlas/lawam.py`(backend `LatentWorldPolicyBackend`)。

## 维度基线
| 量 | 值 | 来源 |
|---|---|---|
| 视觉 backbone | DINOv3-**vitb16**(=DINOv3-base) | yaml `vision_model_id` |
| DINOv3 特征维 `input_dim` | **768** | vjepa_encoder |
| grid | 256×256/16 → **16×16=256 tokens** (K) | yaml |
| LAM `code_dim` | **32** | yaml |
| VLM hidden(Qwen3VL) | 2048 | yaml `vlm_dim` |

## LAM 暴露给 backend 的 3 个入口(不走 forward)
LAM 加载后**全冻结 + eval**。backend 只调:

1. **`extract_vision_features(videos, n=-2)`** → `[B,T,K,D]=[B,T,256,768]`
   - 输入 `[B,T,C,H,W]`(训 T=2)/`[B,C,H,W]`(推理 T=1)。= DINOv3 penultimate grid(去 1CLS+4reg,LN)。
   - backend 取 `h_t=feat[:,0]`(当前)、`h_t1_gt=feat[:,-1]`(未来 GT)。
2. **`decoder(features, actions)`** → `[B,1,256,768]`(仅 `future_prediction=True`)
   - 输入 `features=h_t[B,256,768]` + `actions=pred_action_emb[B,1,32]`(query 维必须=1)。
   - 输出 = **预测的未来视觉特征** `h_t1_pred`,同 DINOv3 空间。→ `loss_perceptual=0.1·MSE(h_t1_pred,h_t1_gt)`。
   - **★ 这就是世界模型预测的位置 ★**
3. **`get_latent_action(videos,dec_videos,embodiment_ids,...)`** → `dict["quantized"]=[B,1,32]`(仅 `enable_loss_distill=True`)
   - (t,t+Δ)一对帧 → 量化 latent-action code。蒸馏 target,teach `vlm_to_lam`。

## 注入链路
- prompt 插 8 个 `<ACT_PH>`(act)+ 8 个(flow)→ backend 用可学 query 替换其 embedding。
- VLM 前向 → 抽 act 占位符 hidden `h_act[B,8,2048]`。
- **`vlm_to_lam`**(QFormer, 单 query)：`h_act[B,8,2048]` → `pred_action_emb[B,1,32]`。
  - 用途① 蒸馏对齐 LAM teacher 的 quantized；② 作 `decoder` 的 action 条件。
- **flow head(action expert)** 吃：`h_t[B,256,768]` + `h_t1_pred/gt[B,256,768]`(未来条件) + 整段 `h_vlm[B,L,2048]` → cross-attn 出动作 `[B,horizon,32]`。

## 开关
- `future_prediction=True`: 调 decoder 出 h_t1_pred + perceptual loss;推理只看当前帧靠 decoder 预测未来。
- `detach_future_feature=True`: flow 梯度不回未来分支(未来只由 perceptual/distill 训)。
- `enable_loss_distill=True`: `vlm_to_lam` 输出模仿 LAM code。
- `loss = loss_flow + 0.1·perceptual + 0.1·distill`。

## Freeze(SFT)
- **LAM 冻**,但 **`lam.decoder` 解冻训练**(lr 组 `policy_backend.lam.decoder`)。
- VLM 前16层+最后层+embed 冻;vision backbone/merger 训。
- **flow head 训**;**`vlm_to_lam` + 两个 query 训**(不在显式 lr 组,需核对 optimizer 覆盖)。

## LMWM adapter 最小契约
替换 `load_latent_action_model(...)` 返回的对象。需:
- **属性**:`.code_dim`(32)、`.input_dim`(768)、`.encoder.grid_height/width`(16,16)、`.encoder.num_frames`(2)、`.decoder`(nn.Module)、标准 `.parameters()/.eval()/.train()`。
- **方法**:`extract_vision_features`(C1)、`decoder`(C2)、`get_latent_action`(C3)。
- **两级集成**:
  - **最小可跑**:`future_prediction=False`+`enable_loss_distill=False` → 只需 C1 + 属性;LMWM 退化为纯特征提取器(h_t1_pred=h_t)。**丢了 LMWM 的预测价值,仅验证接线**。
  - **完整(我们要的)**:实现 **C2 `decoder`=LMWM 预测 next-milestone 特征**(替 LaWM 的 next-frame),C1 保持 DINOv3-vitb16。可选 C3(否则关 distill)。→ **唯一变量=世界模型预测目标(next-milestone vs next-frame)**。

## 待实跑确认
1. `vlm_to_lam`/query 是否真进优化器(不在显式 lr 组)—— 打印 param_groups 核对。
2. `flow_action_query` 的 8 token hidden 未被 flow 显式读取(仅增大 VLM 可训表示)。
3. DINOv3 去前 5 token(1CLS+4reg),换 backbone 注意 token 布局。
