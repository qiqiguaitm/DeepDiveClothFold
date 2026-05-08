#./start_scripts/start_autonomy.sh --execute config_name:=pi05_stand_box_kai0_allgood_25k
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_stand_box_kai0_allgood_25k/t10_allgood_25k/24999 prompt:='stand up the fallen box'

# ★ 当前激活: Task_P Stage 2 ckpt — 默认 RTC ON 模式 ★
# 配置: gf1 bs=128, 8k 配置提前停 (overfit @ step 4000+), ema_decay=0.999, peak_lr=2.5e-5
#
# ckpt 对比:
#   A. step 3000: best val MAE@1=0.0206 (比 Stage 1 终值 0.0362 好 43%)
#   B. step 7999: lowest train loss=0.0009 (但 val MAE@1=0.0219, 轻微过拟合)
#
# 切换 ckpt: 注释/取消注释下方 A 与 B 块。
# 切换 RTC:  启动后在第 2 终端运行 ./start_scripts/rtc_apply.sh off|on|rtc5|...
#            (不要在同一脚本里同时启动 + 改 RTC)

# 选 A: step 3000 (best val MAE, MAE@1=0.0206) — 待用
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_pick_place_box_kai0_unfreeze_8k \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_pick_place_box_kai0_unfreeze_8k/p_unfreeze_8k_v1/3000 \
#  prompt:='pick and place in box'

# 选 B: step 7999 (lowest train loss=0.0009, but overfit: val MAE@1=0.0219, 6% worse than 3000)
# 做 "train loss 最低 vs val MAE 最低" A/B 对照测试。
# 注释掉上面的 A，取消下面 B 的注释即可切换。
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_pick_place_box_kai0_unfreeze_8k \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_pick_place_box_kai0_unfreeze_8k/p_unfreeze_8k_v1/7999 \
#  prompt:='pick and place in box'

# 选 C: Task_P Stage 2 ext — unfreeze_20k/v1 @ step 4000 (gf1 训练, 2026-04-23)
# - 20k-step 训练 schedule (LR 更长衰减, 8k schedule 的升级版)
# - 默认 "best MAE" 候选: step 4000 (按同系列 Stage2 8k 规律推断 4k 区域过 elbow)
# - ckpt 路径是 symlink: kai0/checkpoints/.../4000 → /data1/DATA_IMP/KAI0/ckpt_downloads/pi05_pick_place_box_unfreeze_20k_v1_step4000
# - 复用 unfreeze_8k config (同 model+同 prompt+同 task init; 推理不关心训练时长/LR schedule).
# - 若推理时报 norm_stats 或 shape mismatch → 说明该 ckpt 存的是不同 unfreeze 配置; 联系我加独立 config.
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_pick_place_box_kai0_unfreeze_8k \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_pick_place_box_kai0_unfreeze_20k/p_unfreeze_20k_v1/4000 \
#  prompt:='pick and place in box'

# 选 D: Task_A FlattenFold AWBC "from_official_mixed" @ step 19999 (gf1 训练, 2026-04-23/24)
# - init from official pi05_base (gs://openpi-assets/checkpoints/pi05_base/params)
# - "mixed" 推测含 dagger/advantage 合并数据 (与 awbc_v2 类似)
# - 必须用 pi05_flatten_fold_awbc_from_official_mixed config: 它设了 asset_id="mixed_1",
#   inference 会加载 checkpoint 自带的 assets/mixed_1/norm_stats.json (训练时用的那份).
#   若误用 pi05_flatten_fold_awbc (无 asset_id), 会 fallback 到 repo_id 绝对路径,
#   最终读到 data/Task_A/advantage/norm_stats.json — 与权重不匹配, 关节会被错误归一化.
# - ckpt 由 gf1→TOS fuse(/transfer-shanghai)→sim01 from_tos_file.py 下传 (~2 min), 解压到 /data1/DATA_IMP/KAI0/ckpt_downloads/..., kai0/checkpoints 下 symlink.
# - prompt 用 awbc_v2 块的 YAML-quoted 包装 ('…' 外 "…" 内), 不然 "Advantage: positive" 里的 ": " 会被 YAML 误解析成 mapping.
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_awbc_from_official_mixed \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_awbc_from_official_mixed/beta_official_v1/19999 \
#  prompt:='"Flatten and fold the cloth"'

# 选 E: Task_A FlattenFold AWBC "mixed_gf0_best_at_4k" (gf0 训练, 2026-04-24)
# - 同系列 "from_official_mixed" 训练家族的 gf0 run, 早停在 best step 4000.
# - tar 源: gf1:/vePFS/tim/workspace/deepdive_kai0_tmp/data/mixed_gf0_best_at_4k.tar (12 GB inference-ready 轻量 ckpt)
#   通路: gf1 sudo cp → /transfer-shanghai/KAI0 fuse (TOS) → sim01 from_tos_file.py (~2 min @ 95 MB/s).
# - 复用 pi05_flatten_fold_awbc_from_official_mixed config (同 asset_id=mixed_1, 同 norm_stats).
# - ckpt 结构比 beta_official_v1/19999 少 train_state/ (只有 params+assets+meta, 够推理不够续训).
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_awbc_from_official_mixed \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/mixed_gf0_best_at_4k \
#  prompt:='"Flatten and fold the cloth"'

# 选 F: Task_A FlattenFold AWBC "mixed_gf0_step12999_final" (gf0 训练, 2026-04-25)
# - 同系列 "from_official_mixed" gf0 run 的 step 12999 final ckpt (训练末端快照, 非 early-stop best).
# - tar 源: gf1:/vePFS/tim/workspace/deepdive_kai0_tmp/data/mixed_gf0_step12999_final.tar (11.6 GB inference-ready).
#   通路: gf1 sudo cp → /transfer-shanghai/KAI0 fuse (TOS) → sim01 from_tos_file.py (~2 min @ 100 MB/s).
# - 复用 pi05_flatten_fold_awbc_from_official_mixed config (同 asset_id=mixed_1, 同 norm_stats).
#   norm_stats.json 从 kai0/data/Task_A/advantage/ copy 进 assets/mixed_1/ (md5 15b04c65...).
# - ckpt 结构同 best_at_4k: 只有 params+assets+meta, 无 train_state/ (够推理不够续训).
# - A/B 用途: 对比 4k (early-stop best) vs 13k (final) 是否过拟合.
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_awbc_from_official_mixed \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/mixed_gf0_step12999_final \
#  prompt:='"Flatten and fold the cloth"'

# 选 G: Task_A FlattenFold AWBC "visrobot01_only_best_step6000" (gf0 训练, 2026-04-25)
# - "visrobot01_only" 子集 run, 早停 best @ step 6000 (仅用 visrobot01 本机采集数据, 不混入其他机器源).
# - tar 源: gf1:/vePFS/tim/workspace/deepdive_kai0_tmp/data/visrobot01_only_best_step6000.tar (11.6 GB inference-ready).
#   通路: gf1 sudo cp → /transfer-shanghai/KAI0 fuse (TOS) → sim01 from_tos_file.py (~2 min @ 100 MB/s).
# - 复用 pi05_flatten_fold_awbc_from_official_mixed config (同 asset_id=mixed_1, 同 norm_stats).
#   norm_stats.json 从 kai0/data/Task_A/advantage/ copy 进 assets/mixed_1/ (md5 15b04c65...).
#   ⚠️ 若 visrobot01_only 训练其实用的是纯本机 norm (非 mixed), 关节可能偏; 出异常时切成其训练 data 路径的 stats.
# - ckpt 结构: 只有 params+assets+meta, 无 train_state/ (够推理不够续训).
# - A/B 用途: 对比 "visrobot01_only" 单源 vs "mixed" 混源训练的推理行为.
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_awbc_from_official_mixed \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/visrobot01_only_best_step6000 \
#  prompt:='"Flatten and fold the cloth"'

# 选 H: Task_A FlattenFold AWBC "visrobot01_only_2k_step1999_gf0" (gf0 训练, 2026-04-25)
# - "visrobot01_only" 短训练 schedule (2k steps), 终点 step 1999 ckpt.
# - tar 源: gf1:/vePFS/tim/workspace/deepdive_kai0_tmp/data/visrobot01_only_2k_step1999_gf0.tar (11.6 GB).
#   通路: gf1 sudo cp → /transfer-shanghai/KAI0 fuse (TOS) → sim01 from_tos_file.py (~2 min @ 100 MB/s).
# - 复用 pi05_flatten_fold_awbc_from_official_mixed config (同 asset_id=mixed_1, 同 norm_stats).
#   norm_stats.json 从 kai0/data/Task_A/advantage/ copy 进 assets/mixed_1/ (md5 15b04c65...).
#   ⚠️ visrobot01_only 训练若用纯本机 stats (非 mixed), 关节可能偏; 出问题时拉训练真实 stats.
# - ckpt 结构: 只有 params+assets+meta, 无 train_state/.
# - A/B 用途: 对比 visrobot01_only 短 schedule (2k @ 1999) vs 长 schedule (best @ 6000) 是否 underfit.
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_awbc_from_official_mixed \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/visrobot01_only_2k_step1999_gf0 \
#  prompt:='"Flatten and fold the cloth"'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0.A 真机对照: mix_vis600 step-vs-corner-grasp 曲线 (debug log 选项 I/O/P/Q)
#   docs/deployment/task_a_real_robot_grasp_corner_debug_log.md §0.A
#   依次跑 step 4000/10000/20000/38000, 各 10 次叠衣服, 记录 corner-grasp 成功率.
#   所有 4 个 ckpt 都用同一个 config + 同一份 norm_stats, 只换 step 数字 — 单变量.
# 共用前提:
#   - config: pi05_flatten_fold_mix_vis600 (config.py:2571 已注册, sim01 端)
#   - dataset 同一份: 310 vis + 145 kai0_base + 145 kai0_dagger = 600 ep
#   - asset_id 默认 = repo_id 绝对路径 → norm_stats 读 kai0/data/.../mix_vis600/base/norm_stats.json
#     (md5 38bff549..., 已落盘)
#   - ckpt 结构: params + assets/(空) + _CHECKPOINT_METADATA, 无 train_state/ (只 inference 用)
# RTC: 默认 enable_rtc:=true; 启动后第 2 终端 `./rtc_apply.sh on` 走默认 16/0.5,
#   想切 paper Table 4 (exec_h=25) 用 `rtc_paper`. 建议每 step 都用 same RTC mode,
#   保证变量唯一是 step.
# 通路: gf 上 tar 跳过 train_state (12 GB / step) → sudo cp → /transfer-shanghai/KAI0 fuse
#   → sim01 from_tos_file.py 并行下三个 step (~6 min @ 100 MB/s 总聚合).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 选 O: mix_vis600 step 4000 (Phase 0.A 早期)
#./start_scripts/start_autonomy.sh --execute enable_rtc:=true config_name:=pi05_flatten_fold_mix_vis600 \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_mix_vis600/mix_vis600_v1/4000 \
#  prompt:='"Flatten and fold the cloth"'

# 选 P: mix_vis600 step 10000 (Phase 0.A 中期)
#./start_scripts/start_autonomy.sh --execute enable_rtc:=true config_name:=pi05_flatten_fold_mix_vis600 \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_mix_vis600/mix_vis600_v1/10000 \
#  prompt:='"Flatten and fold the cloth"'

# 选 Q: mix_vis600 step 20000 (Phase 0.A 中后期)
#./start_scripts/start_autonomy.sh --execute enable_rtc:=true config_name:=pi05_flatten_fold_mix_vis600 \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_mix_vis600/mix_vis600_v1/20000 \
#  prompt:='"Flatten and fold the cloth"'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task_A FlattenFold pure_1200 系列 (mixed_1 init + new_norm + 50k schedule)
#   - 配方: cosine warmup=1k, peak_lr=1.5e-5, decay_lr=1.5e-6, ema=0.9999, batch=128
#   - Init from Task_A/mixed_1/params (MA-merged baseline)
#   - asset_id 默认 = repo_id 绝对 → 推理读 <repo_id>/norm_stats.json
#   - ckpt 结构: 扁平 A 类 (params + assets/empty + _CHECKPOINT_METADATA, 12 GB, 无 train_state)
#   - 通路: gf1 to_tos_file.py → TOS → sim01 from_tos_file.py (~2 min @ 91 MB/s)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 选 R: task_a_pure_1200_new_norm step 49999 (final, gf0 #24 训练完成 2026-05-02)
# - 训练 dataset: kai0/data/Task_A/self_built/A_pure_1200/base
#   1142 train + 58 val (620 visrobot01 originals + 580 hflip mirrors), ALL 8 dates 04-23..04-30
# - inline_eval best @ step 48000=49999 tied: MAE@1=0.0145, @10=0.0255, @25=0.0384, @50=0.0539
# - norm_stats md5: 35a68a83... (与 gf 训练源对齐, 已 scp)
#./start_scripts/start_autonomy.sh --execute enable_rtc:=true config_name:=pi05_flatten_fold_a_pure_1200 \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/task_a_pure_1200_new_norm_best_step49999 \
#  prompt:='"Flatten and fold the cloth"'

# 选 S: task_a_new_pure_1200_new_norm step 38000 (best valid, gf1 #25 训练 step 40000 ckpt save 时 ENOSPC crash)
# ⭐ 当前推荐 (best MAE@1=0.0104, 优于所有 Task A 历史训练)
# - 训练 dataset: kai0/data/Task_A/self_built/A_new_pure_1200/base
#   1143 train + 57 val (613 originals + 530 mirrors), 仅 6 个 -new 日期 (04-23/24/25/28/29/30-new)
# - inline_eval @ step 38000: MAE@1=0.0104, @10=0.0227, @25=0.0378, @50=0.0569
#   (step 40000 inline-eval 实测 MAE@1=0.0103 但 ckpt 没保存全)
# - 数据来源 -new 日期质量更高, 较 a_pure_1200 (8 日期) 同 step 低 ~32%
# - norm_stats md5: 43e8a4ec... (与 gf 训练源对齐, 已 scp)
./start_scripts/start_autonomy.sh --execute enable_rtc:=true config_name:=pi05_flatten_fold_a_new_pure_1200 \
  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/task_a_new_pure_1200_new_norm_step30000 \
  prompt:='"Flatten and fold the cloth"'

# 选 I: mix_vis600 step 38000 (Phase 0.A best MAE @ 0.0146, 当前部署版本)
#./start_scripts/start_autonomy.sh --execute enable_rtc:=true config_name:=pi05_flatten_fold_mix_vis600 \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_mix_vis600/mix_vis600_v1/38000 \
#  prompt:='"Flatten and fold the cloth"'

# 选 J: Task_A FlattenFold "mixed_visrobot01" step 49999 (xyh 训练, 2026-04-25→26 跑完)
# - config: pi05_flatten_fold_mixed_visrobot01 (sim01 config.py:1793 已注册).
#   block 里 num_train_steps=12_000 / batch_size=128, 但实际跑用 CLI override:
#   --batch-size 64 --num-train-steps 50000 --save-interval 5000 → step 编号到 49999 (50k schedule final).
# - 训练数据: kai0/data/Task_A_mixed_gf1/base (visrobot01 + base + dagger 等 N 平衡混合,
#   train_scripts/data/build_task_a_mixed.py 早先版本产物).
# - asset_id 默认 = repo_id (绝对路径), openpi 从 <repo_id>/norm_stats.json 直读;
#   norm_stats 已落到 /data1/tim/.../kai0/data/Task_A_mixed_gf1/base/norm_stats.json
#   (md5 731fb5df..., 与 ckpt 自带那份一致, 来自 xyh 训练时 dataset 侧).
# - run_id: mixed_visrobot01_1500, wandb dnafjz77 (xyh 在外部机器训, ckpt 通过本地拷贝过来).
# - ckpt 结构: params + assets/(空) + _CHECKPOINT_METADATA + train_state/ (42 GB, 包含续训状态).
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_mixed_visrobot01 \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_mixed_visrobot01/mixed_visrobot01_1500/49999 \
#  prompt:='"Flatten and fold the cloth"'

# 选 K: Task_A FlattenFold "mixed_1" 初始化模型 (微调 init source, 项目级 asset)
# - 这是所有 fold/awbc 系列微调的 init weight + 默认 norm_stats 来源, 单独测一下"未微调起点"基线.
# - 路径: kai0/checkpoints/Task_A/mixed_1/  (项目级 asset 目录, 见 README §4.3)
# - 结构: params/ + _CHECKPOINT_METADATA + assets/ + norm_stats.json (md5 b206072c..., 5343 B)
# - config 用 pi05_flatten_fold_awbc_from_official_mixed: 它显式设了
#     AssetsConfig(assets_dir="kai0/checkpoints/Task_A", asset_id="mixed_1")
#   → openpi 直接从 kai0/checkpoints/Task_A/mixed_1/norm_stats.json 加载 (本就在 ckpt 同目录, 0 配置).
# - 用途: 跟微调后的 ckpt (选 D/E/F/I/J) 做 baseline 对比, 看微调到底带来了什么 / 是否退化.
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_awbc_from_official_mixed \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/Task_A/mixed_1 \
#  prompt:='"Flatten and fold the cloth"'

# 选 L: Task_A FlattenFold "pure_vis600" best step 39999 (gf1 训练 2026-04-26→28 跑完)
# - 训练 dataset: kai0/data/Task_A/self_built/pure_vis600/base (309 vis_base ORIGINALS + 291 hflip MIRRORS,
#   left↔right swap aug; zero kai0 source). 40k cosine schedule, peak_lr=1.5e-5, ema=0.9999.
#   Init from Task_A/mixed_1/params. inline_eval best @ step 39999 (final).
# - tar 源: gf:/vePFS/tim/workspace/deepdive_kai0_tmp/data/pure_vis600_best_step39999.tar (11.6 GB).
#   通路: gf1 sudo cp → /transfer-shanghai/KAI0 fuse (TOS) → sim01 from_tos_file.py (~2 min @ 53 MB/s).
# - 用 NEW config pi05_flatten_fold_pure_vis600 (config.py 已注册).
#   asset_id 默认 = repo_id (绝对路径) → openpi 从
#   kai0/data/Task_A/self_built/pure_vis600/base/norm_stats.json 直读 (md5 d8b80670..., 已 scp 落盘).
# - ckpt 结构: params + assets/(空) + _CHECKPOINT_METADATA, 无 train_state/.
# - A/B 用途: 与 mix_vis600 (选 I) 对比, 看 hflip 镜像增广 vs kai0 真数据混合 哪种更好.
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_pure_vis600 \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_pure_vis600/pure_vis600_v1/39999 \
#  prompt:='"Flatten and fold the cloth"'

# 选 M: Task_A FlattenFold "vis_base_40k" best step 36000 (gf 训练 2026-04-26→28 跑完)
# - 训练 dataset: kai0/data/Task_A_visrobot01_only/base (310 vis_base 原始, 288 train+22 val,
#   strict subset of pure_vis600, 不含 mirror, 不含 kai0). 40k cosine, peak_lr=1.5e-5, ema=0.9999.
#   Init from Task_A/mixed_1/params. inline_eval best @ step 36000 (early-stop, 非 final).
# - tar 源: gf:/vePFS/tim/workspace/deepdive_kai0_tmp/data/vis_base_40k_best_step36000.tar (11.6 GB).
#   通路: 同选 L (TOS bridge).
# - 用 NEW config pi05_flatten_fold_vis_base_40k (从 gf port 到 sim01 config.py).
#   asset_id 默认 = repo_id → 读 kai0/data/Task_A_visrobot01_only/base/norm_stats.json (md5 01842ddc..., 已 scp).
# - ckpt 结构: params + assets/(空) + _CHECKPOINT_METADATA, 无 train_state/.
# - A/B 用途: 与 pure_vis600 (选 L) 对比, 看 hflip 镜像有没有帮助 (相同 base data, 唯一变量是 +mirrors).
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_vis_base_40k \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_vis_base_40k/vis_base_40k_v1/36000 \
#  prompt:='"Flatten and fold the cloth"'

# 选 N: Task_A FlattenFold "mix_apr28_450_inherit_norm" best step 28000 (gf 训练 2026-04-29→30)
# - 训练 dataset: kai0/data/Task_A/self_built/mix_apr28_450_inherit/base
#   data/videos symlink 自 mix_apr28_450/base (450 ep mix: 150 vis_2026-04-28 + 150 kai0_base + 150 kai0_dagger).
#   30k cosine, peak_lr=1.5e-5, ema=0.9999. Init from Task_A/mixed_1/params.
# - 与 mix_apr28_450 (NEW_NORM) 的对照: dataset 完全相同, 唯一差异 = norm_stats 来源.
#     INHERIT_NORM (本): norm_stats.json 直接复用 Task_A/mixed_1 那份 (md5 b206072c…).
#     NEW_NORM (对照):   norm_stats.json 从当前 405 train 重新算.
#   Tests whether "用 init 模型的旧 stats" 比 "重新算" 表现差/好/无差.
# - tar 源: gf:/vePFS/tim/workspace/deepdive_kai0_tmp/data/mix_apr28_450_inherit_norm_best.tar (11.6 GB).
#   通路: gf sudo cp → /transfer-shanghai/KAI0 fuse (TOS) → sim01 from_tos_file.py (~2 min @ 95 MB/s).
#   step 28000 通过比对 _CHECKPOINT_METADATA commit_timestamp 与 gf ckpt 目录推断得出.
# - 用 NEW config pi05_flatten_fold_mix_apr28_450_inherit_norm (sim01 config.py 已 port).
#   asset_id 默认 = repo_id (绝对路径) → openpi 从
#   kai0/data/Task_A/self_built/mix_apr28_450_inherit/base/norm_stats.json 直读
#   (md5 b206072c…, 与 mixed_1 那份完全相同, 已 rsync 落盘).
# - ckpt 结构: params + assets/(空) + _CHECKPOINT_METADATA, 无 train_state/. 够推理不够续训.
# - RTC: 启动后第 2 终端跑 `./start_scripts/rtc_apply.sh rtc_paper` 以对齐 Table 4
#   (exec_h=25, max_guid=0.5; 见 logs/rtc_config_compare_2026-04-29.md). 若 0.5 偏弱
#   再切 `rtc_paper_strong` (max_guid=5.0). enable_rtc 这里设 true 以启用 guidance.
#./start_scripts/start_autonomy.sh --execute enable_rtc:=false config_name:=pi05_flatten_fold_mix_apr28_450_inherit_norm \
#  checkpoint_dir:=/home/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_mix_apr28_450_inherit_norm/mix_apr28_450_inherit_norm_v1/28000 \
#  prompt:='"Flatten and fold the cloth"'

# Task_E vision-unfreeze full-param @ step 1999 (gf1 bs=128, 2000 steps, 2026-04-22)
# inline_eval EMA MAE@1=0.0396 (step 1999). Saved params are EMA-weighted (~18% trained + 82% init).
# Note: 保存的 params = EMA params; live params 在 train_state 不单独提取。
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_stand_box_kai0_unfreeze_2k \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_stand_box_kai0_unfreeze_2k/unfreeze_2k_v1/1999 \
#  prompt:='stand up the fallen box'

# Task_P vision-unfreeze full-param @ step 1999 (gf0 bs=128, 2000 steps, 2026-04-22)
# inline_eval EMA MAE@1=0.0362 (step 1999). vs P-T10 baseline 0.0633: 43% better (EMA).
#./start_scripts/start_autonomy.sh --execute #config_name:=pi05_pick_place_box_kai0_unfreeze_2k \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/#pi05_pick_place_box_kai0_unfreeze_2k/p_unfreeze_2k_v1/1999 \
#  prompt:='pick and place in box'

# awbc_v1 baseline (binary prompt) - joint_1=0.0050 @ step 20,000
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_flatten_fold_awbc
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/#pi05_flatten_fold_awbc/gf0_awbc_baseline_v2/20000 prompt:='Flatten and fold the #cloth. Advantage: positive'

# awbc_v2_vanilla (dagger + mirror, 温和 aug) - joint_1=0.0048 @ step 29,000 (29999 = 实际最优可用点)
# NOTE: prompt is wrapped in YAML-string quotes (outer '…', inner "…") because the ": " in
# "Advantage: positive" otherwise parses as a YAML mapping → launch rejects dict for a str param.
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_flatten_fold_awbc_v2 \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_awbc_v2/gf1_awbc_v2_vanilla/29999 \
#  prompt:='"Flatten and fold the cloth. Advantage: positive"'

# awbc_v2_robust (dagger + mirror, 激进 aug) - joint_1=0.0051 @ step 29,000
#./start_scripts/start_autonomy.sh --execute config_name:=pi05_flatten_fold_awbc_v2_robust \
#  checkpoint_dir:=/data1/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_awbc_v2_robust/gf0_awbc_v2_robust_v1/29999 \
#  prompt:='"Flatten and fold the cloth. Advantage: positive"'

# ─────────────────────────────────────────────────────────────────────
# RTC (Real-Time Chunking) — 运行时调整推理频率，不改源码
# 默认: inference_rate=3Hz (每 333ms / ≈10 publish-step 重查策略)
# 在第二个终端运行以下命令改变 RTC 粒度（autonomy 启动后）:
#
#   cd /data1/tim/workspace/deepdive_kai0
#   source ros2_ws/install/setup.bash       # ros2 命令可用
#   ./start_scripts/rtc_apply.sh show       # 查看当前值
#   ./start_scripts/rtc_apply.sh rtc5       # 每 5 步 replan (推荐测试)
#   ./start_scripts/rtc_apply.sh rtc3       # 每 3 步 replan (激进)
#   ./start_scripts/rtc_apply.sh default    # 恢复默认
#
# 对 "抓取瞬间偏" 失败模式，rtc5/rtc3 预期改善 15-30%.
# 实时改参无需重启 policy，可 A/B 对比 rollout.
