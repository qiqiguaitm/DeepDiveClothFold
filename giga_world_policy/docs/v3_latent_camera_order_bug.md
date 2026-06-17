# v3 latent 相机拼图错乱 bug —— 排查与修复全记录

> 2026-06-17。fastwam-v5(独立 ActionDiT @ v3)训练不收敛的根因排查、像素级确认与修复。
> 连带发现 gwp_abs_v4 也踩了同一个坑(只是症状被掩盖)。wam_fold_v1 不受影响。

---

## 0. TL;DR

- **现象**:fastwam-v5 在 v3 数据上训练,`loss_action` 正常下降,但离线评测 **MAE@48 卡在 ~0.18 不收敛**(@1 却在降)。
- **根因**:fastwam `scripts/compute_latents.py` 用 `sorted()` **按字母序**自动探测相机。v3 相机名 `top_head/hand_left/hand_right` 字母序 = `[hand_left, hand_right, top_head]` → `window_pixels` 把 **CAMS[0]=hand_left(腕部)当 256×320 主图,把 top_head(俯视全局,叠衣最关键)压进 128×160 腕部小槽(降采样 4×)**。
- **为何只 @48 崩**:fastwam 从**单帧观测**预测 48 步动作链。训练用错拼图的 latent、评测(`eval_offline_fold.py` prep_image)用正确顺序 → 训练/评测画面不一致 → 依赖视觉的长程预测崩,靠 proprio 锚定的近程(@1)还撑得住。
- **确认**:A/B 实测 + VAE 解码像素双重确认(见 §3)。
- **影响**:**v3 受影响**(fastwam-v5 + gwp_abs_v4,均读这份 latent);**v1 不受影响**(`cam_high` 字母序恰好排第一 = 俯视图,巧合正确)。
- **修复**:compute_latents 改角色感知排序;重算 `vae_latent_v3fix`;两个模型用修正 latent 重训。

---

## 1. 现象(symptoms)

fastwam-v5 = fastwam-v4 配方(独立 ActionDiT, LR 1e-4 cosine, batch 16)换 v3 数据,50k 步。eval 每 2500 步:

| step | @1 | @24 | @48 |
|---|---|---|---|
| 2500 | 0.0330 | 0.1623 | 0.2075 |
| 7500 | 0.0255 | 0.1208 | 0.1787 |
| 12500 | 0.0168 | 0.1232 | 0.1824 |

- `loss_action` 持续降(10→2→0.06),@1 持续降(0.033→0.017),但 **@48 卡在 0.17-0.20 不动**。
- 对比 fastwam-v4(v1)同 step @48 已 0.106 并继续下行;v5 @48 起点(0.208)比 v4 同 step 高 57%。
- "训练 loss 降 / @48 不降 / 近端好远端崩" —— 典型的**评测端系统性偏差**或**视觉条件错位**信号。

## 2. 排查过程(falsification chain)

1. **排除架构差异**:先怀疑独立 ActionDiT vs 共享 transformer。但 fastwam-v4(v1)同架构收敛良好(0.091)→ 不是架构。
2. **排除相机键顺序(config 层)**:v3 data config 相机顺序 = `top_head, hand_left, hand_right`(正确),`concat_multi_camera=robotwin`,与 eval prep_image 的拼图布局一致 → config 层没问题。
3. **排除 latent 格式**:v1 与 v3 latent 同格式 `{starts, latents(N,48,4,24,20), stride}` → 不是格式问题。
4. **定位到拼图生成代码**:fastwam `compute_latents.py:main` 用 `sorted(detected)` 字母序探测相机,而 `window_pixels` 按位置拼图(`per_cam[0]`→256×320 主图)。v3 字母序把 hand_left 顶到主图位、top_head 压进小槽。
5. **解释 gwp 为何"没事"**:gwp build_ref_image 是 3 等宽横排(与 fastwam 拼图不同);一度以为 gwp 用不同 latent。但代码核查发现 **gwp 经 lerobot_dataset 的 fastwam-format 分支读的就是这份 fastwam latent** → gwp 也在错拼图上训练,只是症状被掩盖(三路相机都拍桌面,腕部相机在主图位仍提供大量信息,故仍收敛到 0.104)。

## 3. 双重确认(代码 + 像素)

### A/B 实测(eval 顺序 vs 训练 latent 顺序)
v5 step12500,2 ep:

| eval prep_image 相机顺序 | @48 |
|---|---|
| `top_head,hand_left,hand_right`(正确顺序,**与训练 latent 失配**) | **0.2086** |
| `hand_left,hand_right,top_head`(**匹配训练 latent 的错乱顺序**) | **0.1444** |

→ 让 eval 匹配(错乱的)训练 latent,@48 骤降 31% 并贴近 gwp 同步 0.142 → 坐实是 latent/eval 拼图错位,模型本身在学。

### VAE 解码像素(把 gwp_abs_v4 实际训练 latent 解回像素)
解码 `vae_latent/episode_000000.pt` 窗口0 → 像素,对比三路原始相机帧:

| | 256×320 主图 = 哪个相机 | top_head(俯视)位置 |
|---|---|---|
| **OLD(gwp_abs_v4 / fastwam-v5 训练用)** | **hand_left(腕部)** ❌ | 压进 128×160 小槽,降采样 4× |
| **NEW(vae_latent_v3fix 修正后)** | **top_head(俯视)** ✅ | 全分辨率主图 |
| **v1(对照)** | **cam_high(俯视)** ✅ | 全分辨率主图(字母序巧合正确) |

像素铁证:OLD 主图 == 原始 `hand_left`(广角桌面+顶部白布+双夹爪);NEW 主图 == 原始 `top_head`(俯视单布居中)。

## 4. 影响范围

| 数据集 | 相机名 | 字母序结果 | 主图是否俯视 | 受影响? |
|---|---|---|---|---|
| **wam_fold_v1** | cam_high/cam_left_wrist/cam_right_wrist | [cam_high, ...] | ✅ 是(cam_high) | **否**(巧合正确) |
| **wam_fold_v3** | top_head/hand_left/hand_right | [hand_left, hand_right, top_head] | ❌ 否(hand_left) | **是** |

- 受影响:fastwam-v5、gwp_abs_v4(均读 v3 `vae_latent`)。
- 不受影响:fastwam-v4(0.091)、delta-5x、abs_50k(均 v1,latent 正确)。

## 5. 修复

1. **`fastwam/scripts/compute_latents.py`**:字母序 `sorted()` → **角色感知排序**(top/head/high/overhead 优先当主图,再 left,再 right)+ 新增 `--cameras` 显式覆盖 + `--out_dir`。验证:v3 → `[top_head, hand_left, hand_right]`,v1 → `[cam_high, cam_left_wrist, cam_right_wrist]`。
2. **重算 latent**:AIHC 1×8 A100 job(`run_compute_latents_v3fix_aihc.sh` / `aijob_compute_latents_v3fix.json`),`--cameras top_head,hand_left,hand_right`,写**新目录** `vae_latent_v3fix`(不动旧 `vae_latent`,gwp_abs_v4 用过)。
3. **eval 适配**(`eval_offline_fold.py`,已向后兼容 v1):env 覆盖 `EVAL_VAL_ROOT/EVAL_VIEW_KEYS/EVAL_DATA/EVAL_TASK/EVAL_TEXT_EMB`;prep_image 按 VK 位置取相机(不再硬编码 v1 名)。eval 顺序 `top_head,hand_left,hand_right` 现与修正 latent 一致。
4. **config 指向修正 latent**:fastwam `configs/data/visrobot01_v3_fold.yaml` latent_cache_dir → `vae_latent_v3fix`;gwp 新配置 `visrobot01_gwp_abs_v5.py`(继承 v4,latent_dir → `vae_latent_v3fix`)。
5. **`cmp_report.py`**:槽位中性化 baseline/target(旧 --delta_run/--abs_run 仍作别名)+ --label_baseline/--label_target/--title/--desc;fastwam-v5 报告 = target:fastwam-v5(独立ActionDiT@v3) vs baseline:gwp_abs_v4(共享transformer@v3),同 v3_val。

## 6. 重训与对照

用修正 latent `vae_latent_v3fix` 各重训一版:
- **gwp_abs_v5**:`visrobot01_gwp_abs_v5.config`(共享 transformer,修正 latent)。
- **fastwam-v5**(重提):`visrobot01_v3_fold_1e-4` + config 指 vae_latent_v3fix。

预期:俯视图全分辨率当主图后,两者 @48 应较各自"错拼图"基线下降;之后可做**唯一变量=架构**(独立 ActionDiT vs 共享 transformer @ 同修正 v3 latent + 同 v3_val)的干净对比。

## 7. 经验教训

- **自动探测顺序绝不能用字母序**:相机/通道顺序必须显式或角色感知;字母序在 v1 偶然对、在 v3 翻车。
- **latent 缓存要能解码回像素核验**:shape 相同不代表内容/布局相同(v1/v3、不同拼图同 shape `(48,4,24,20)`)。
- **"近端好/远端崩"是视觉条件错位的指纹**:单帧→长链预测里,长程更依赖视觉,故对 latent 拼图错位最敏感。
- **共享资源跨工具复用要查格式来源**:`vae_latent` 由 fastwam 写、gwp 经兼容分支读 —— 一个 bug 同时污染两个项目。
