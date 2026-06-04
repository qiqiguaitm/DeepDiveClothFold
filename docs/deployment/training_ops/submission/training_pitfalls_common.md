# 训练提交 — 跨集群共性踩坑 (Common Pitfalls)

> **作用**: 汇总**与集群无关、任何训练提交都会踩**的坑 (数据/init/eval/config)。集群**特有**的坑见各自文档:
> - Volc (cnbj/cnsh): [`volc_ml_platform.md`](volc_ml_platform.md) §"Volc 特有踩坑"
> - uc (uc01/02/03): [`uc_cluster_jobs.md`](uc_cluster_jobs.md) §12.10 / §12.11
>
> 提交前还要过 [`README.md`](README.md) 的 Pre-Submit Checklist。本文是 checklist 的"为什么 + 怎么修"展开。
> **建立**: 2026-06-01 (从多轮迁移实战归纳)。

---

## 1. norm_stats — 不自动算, 且脚本/参数易错 ⚠️

> ✅ **2026-06-03 起部分缓解**: 构建脚本 `train_scripts/kai/data/build_no_release.py` **构建完成后已自动从刚建好的数据集重算 norm_stats**(helper `norm_stats_from_dataset.py`,与下方 `compute_norm_states_fast.py` 数值一致、config-free,默认开,`--no-norm-stats` 关)。**用该脚本 + kai0 venv 构建的 kai pi0/pi05 数据集无需再手动算**。下述手动流程仍适用于:① 其它未接线的构建脚本;② EE6D/xvla 集(不适用,靠 loss-scale);③ 校验/重算。

- **`train.py` 不算 norm_stats** (只 `shutil.copy`)。提交前必须在**数据所在机器**手动算(或用上面已自动化的构建脚本)。
- **用 `compute_norm_states_fast.py --config-name <config>`** (带 `_fast`, base_dir 直读绝对路径)。
  - ❌ 不要用 `compute_norm_stats.py` —— 它把绝对 repo_id 当 HF repo → `HFValidationError`。
  - ❌ 不要漏 `--config-name` flag (tyro 要 flag, positional 会打 helptext)。
- **跨集群路径**: `compute_norm_states_fast.py` 的输出落到 config 的 `repo_id` 路径。若 config repo_id 是别的集群的绝对路径 (如 cnsh `/vePFS/...`) 而你在 cnbj/uc 跑 → 写入会权限拒/落错位。
  - **绕过**: 在数据所在集群算一次, 然后把 `norm_stats.json` 经 TOS/scp 拷到目标集群同名数据目录 (同源数据 norm_stats 一致)。

## 2. 绝对路径 repo_id — 新版 huggingface_hub 拒绝 ⚠️

- config 里 `repo_id` 写**绝对本地路径** (`/vePFS/.../A_xxx`) 时, 新版 lerobot/hub (≥0.30, 如 cnbj py3.12 venv hub 0.32) **严格校验 repo_id 格式 → `HFValidationError: Repo id must be in the form repo_name`**。旧 venv (cnsh py3.11) 放行, 所以"同款 config 在沪能跑、在京崩"。
- **已修** (`data_loader.py`, commit `0484255`): 检测绝对路径 → 走 `root=` 参数 + dummy `local/<name>` label。**确保目标集群 git pull 到含此 fix 的 commit**。
- **跨集群提交**: config repo_id 常是某一集群的路径。在别的集群跑要 **CLI override `--data.repo_id <目标集群路径>`** (见各集群 YAML 的 `--data.repo_id` 用法)。

## 3. 自建数据集构建 — 两个必检 ⚠️

- **视频目录命名用 feature key, 不用裸 cam 名**: lerobot `video_path` 模板是 `videos/chunk-000/{video_key}/...`, `video_key = observation.images.top_head`。若 build 脚本写成裸 `top_head/` → lerobot `get_episodes_file_paths()` 的文件存在 assert 失败 → 回退 HF → `OfflineModeIsEnabled`。(build_no_release.py commit `89f43f5` 已修)。
- **`info.json` 的 `chunks_size` 必须 ≥ `total_episodes`(>1000 ep 数据集致命)** ⚠️: 自建脚本把所有 ep 写进单个 `data/chunk-000/`, 但 `chunks_size` 从源 info 抄来常是 **1000**。lerobot 用 `episode_chunk = ep_idx // chunks_size` 算路径 → ep≥1000 期望 `chunk-001/episode_001000.parquet`(不存在)→ 文件 assert 失败 → `get_safe_version` 打 HF hub → `OfflineModeIsEnabled` 崩。**症状极隐蔽**: <1000 ep 数据集(如 955)完全正常, 一旦 ≥1000(1033/1940)就崩; 按 `episode_chunk=0` 写的体检脚本**查不出来**(必须用真实 `ep//chunks_size`)。判据: `info.chunks_size` vs `total_episodes`, 且 `ls data/` 只有 `chunk-000`。修复无需重建: `info["chunks_size"] = max(1000, N)`(单 chunk 布局下 chunks_size≥N 即合法)。(build_no_release.py / build_smooth800_dagger.py commit 已修; 2026-06-04 Exp-C 1940ep + dagger-A 1033ep 实例)。
- **`info.json` 的 `total_episodes`/`total_videos`/`splits` 必须 = 实际写入 ep 数, 不是请求数** ⚠️: build 脚本若**跳过坏视频 ep**(如 vis_base restructure 留下的 broken symlink), 写入数 < 请求数。若 info 用请求数(pre-skip)→ 多出幽灵尾索引 `episode_00045X` → 同上 lerobot 文件 assert 失败 → `get_safe_version → list_repo_refs` 打 HF hub → `OfflineModeIsEnabled` 崩 (与上一条**同症不同因**)。判据: `info.total_episodes` vs `wc -l meta/episodes.jsonl` vs `ls data/chunk-000/*.parquet | wc -l` **三者必须一致**。修复无需重建: 按 `episodes.jsonl` 行数 patch info.json 的三字段即可 (files 本就连续 0..N-1)。(build_smooth800_dagger.py commit `0daee64` 已修: 用 `new_idx` 非 `len(all_eps)`)。
- **`.kai0_ts_validated` marker**: kai0 patch 的 lerobot 靠它跳过 timestamp 校验。自建集没有 → 首次 load 会做 timestamp 检查 (慢但不致命)。可 `touch .kai0_ts_validated` 跳过 (确认数据对齐后)。
- 其它 (parquet 7 标准列 / episode meta 完整) 见 `uc_cluster_jobs.md §12.8`。

## 4. init ckpt 完整性 — 按 SIZE 校验, 不按文件数 ⚠️

- TOS / 跨集群同步 init (orbax params) **易截断**: 下载不全 → 训练读权重时 `FAILED_PRECONDITION: Truncated Zstd-compressed stream`。
- **文件数对 ≠ 完整**: 截断时 `params/d/<chunk>` 可能只几 KB。**按 size 校验**: 完整 pi05 `params/ocdbt.process_0` ≈ **22G** (含全权重)。`du -sh params/ocdbt.process_0` 不到 GB 级 = 截断, 重传。
- 重传用 `tosutil cp -r -f -j 8 -p 4` (force + 并行), 完成后 `du -sh` 确认。

## 5. TOS cp 路径嵌套 ⚠️

- `tosutil cp -r tos://.../mixed_1_clean/ dst/mixed_1_clean/` → 产生 `dst/mixed_1_clean/mixed_1_clean/params` (多一层)。
- **修复**: 同步后 `[ -d $B/mixed_1_clean/params ] && mv $B/mixed_1_clean/* $B/ && rmdir $B/mixed_1_clean`。
- 或 dst 末尾不带同名目录: `cp -r tos://.../mixed_1_clean/ dst/` (落到 `dst/mixed_1_clean/`)。

## 6. eval 脚本 prompt 默认值是错的 ⚠️ (offline MAE 致命)

- `train_scripts/kai/eval/eval_val_action_mse.py` 的 `--prompt` **默认 `"stand up the fallen box"`** (Task E)。评 Task A 叠衣模型若不传 prompt → 模型被喂错指令 → **MAE 全程虚高 ~1.8×** (早期 step 虚高更多, 模型对语言更敏感)。
- **必须传** `--prompt "Flatten and fold the cloth."` (或对应任务 prompt)。
- 教训: 2026-05-31 pure_200 PyTorch eval 因此误得 @50=0.0646, 正确 prompt 实为 0.0350。

## 7. inline-eval silent failure ⚠️

- config `inline_eval_val_root` 路径 stale (如数据迁移后没更新) → `train.py` 防御捕获为 warning, 训练继续但**这些 step 无 MAE** (silent eval=0)。
- 提交前 verify val 路径存在 (YAML pre-flight 加 `[ -f "$VAL/meta/episodes.jsonl" ] || exit 13`)。
- 补救: 对 kept ckpt offline eval 重建曲线 (`eval_val_action_mse.py`, 记得带正确 prompt 见 §6)。
- ⚠️ **数据集被还原成 LFS 指针残桩**: offline eval 报 `FileNotFoundError: episode_0000XX.parquet` 但文件"存在"——`stat` 显示 ~112B(parquet)/ ~119B(mp4)= **git-LFS 指针 stub, 非真数据**。训练时是真数据, 之后被 `git reset --hard`(cron 同步)还原成仓库里 commit 的 LFS 指针。**判据**: `stat -c %s` 整套 parquet 都 112B / mp4 都 119B = 全被 stub。**修复**: 从有真数据的副本(如 uc01 NFS)relay 回来覆盖, `head -c4 *.parquet` 应为 `PAR1`。(2026-06-02 vis_v2_merged_val 实例)

## 8. config 必须先 commit+push (gf3/uc cron pull) ⚠️

- gf3 + uc01/02/03 由 1-min cron `reset --hard origin/main` 镜像 main。改 config/代码后**必须在 gf0 `git push`**, 等 ~1min 目标机 pull 到再提交。
- 别直接在 gf3/uc 改代码 (下次 reset 覆盖)。验证: `ssh <host> "cd repo && git log --oneline -1"` 看 HEAD 是否含你的 commit。
- 提交 volc 报 `Config 'X' not found` = 目标集群 vePFS checkout stale, pull 一下。

## 9. 多机训练 ckpt 必须落到所有节点可见的同一共享盘 ⚠️ (多机专属, 单机无此坑)

- 多机 orbax `CheckpointManager` 要求**所有进程写同一个物理共享目录**: primary(proc0)建 `array_metadatas` 等目录, 其余进程跨盘等它出现。若 ckpt-base-dir 解析到**各节点本地盘**(每台一块不同物理盘), proc1 永远等不到 proc0 的目录 → `Timed out waiting for array_metadatas base directory creation (timeout=600s)` → `Shutdown barrier` → 全崩, **无任何 finalized ckpt**。
- 典型陷阱: `kai0/checkpoints` 在某些机器是 **symlink → 节点本地 SSD**(为单机加速, 见 uc `local_ckpts` / sim01 `/data1`)。**单机训练正好要这个本地盘; 多机训练这恰恰是致命的**。
- **修复**: 多机一律显式 `--checkpoint-base-dir <真共享 FS>`(uc: workspace NFS `/data/shared/ubuntu/workspace/multinode_ckpts`; volc: vePFS 本就是共享, 默认 `kai0/checkpoints` 即可)。
- **稳定判据(关键)**: `Step N: loss` 同步下降**只证明 NCCL/前反向通, 不证明能落盘**。多机真正过关 = **熬过第一次 ckpt save**(看到 finalized `<step>/` 目录、非 `*.orbax-checkpoint-tmp-*`, 且训练继续)。详见 [`uc_cluster_jobs.md §12.11 坑 9`](uc_cluster_jobs.md)。

## 10. 任务"实例异常结束"但**无 entrypoint 日志** = 死在 pod 启动层 ⚠️

- 现象: volc job `Failed`, Message `worker-N 共 X 实例异常结束`, **`StartTime` 空, 且 vePFS 上无该 STAMP 的训练日志文件**。意味着 entrypoint 在 `exec >> "$LOG"` 重定向**之前**就死了 → 不是数据/config/代码问题(那些在重定向之后, 会写进日志), 而是**镜像拉取 / vePFS 挂载 race / 调度** 层(尤其队列拥挤时)。
- **API 取不到 pod stdout**(`GetJob` 只有 `Status`; `mlp` CLI 多数机器没装)。所以要**自证**: entrypoint 第一步(redirect 前)写一个 **vePFS breadcrumb** `echo ... > $LOG_DIR/preflight_<exp>_${STAMP}.txt`(跳板机可读)。复跑后 breadcrumb 在 = 挂载 OK、死在更后; breadcrumb 也没有 = 挂载/pod-init 层。
- **缓解**: `RetryOptions: {EnableRetry: true, MaxRetryTimes: 1}` 让瞬时节点/挂载故障自愈; 多 worker 时每个 worker 用 `_node${NODE}` 区分日志名(别全复用一个名)。
- ⚠️ **真凶常是"镜像在节点上没缓存 → 冷拉把 2 节点 gang staging wedge 住"**(2026-06-04 cnbj 实测确认): 控制台 `State: Staging` + `UpdateTime` 冻结 1h+ + `异常事件 0` + `0/2 实例` = 卡在平台备容器层, 既不报错也不前进; `RoleMinReplicas:2` gang 要两个 worker 同时起来, 冷拉镜像时极易卡死(单节点 job 无此问题, 所以 cnsh 单机几秒就 Running)。**判据**: Staging 态 + UpdateTime 长时间不变 + 无 entrypoint breadcrumb。**修复**: 换一个**该集群节点上已缓存/更小的镜像**——h2r:1.0(visincept registry)换成 `dvs-cr-cn-beijing.cr.volces.com/vis_robot/kai:kai0-gf1` 后, canary submit→Running **20 秒**(原镜像卡 1h22m)。镜像只提供基础 OS/CUDA(venv 在 vePFS), 故换镜像零成本。同源解释了第一轮"实例异常结束无日志"(冷拉 backoff 到超时变 Failed)。先单任务验证再全量 stop+重提(commit `3782b06`)。

## 11. 被 import 的 working-tree 文件含 **git 冲突标记** → 全任务 `SyntaxError` ⚠️

- volc/单机任务直接 `cd <repo> && python scripts/train.py` 跑的是**共享盘上的活 working tree**。若该 tree 里某个被 import 的 .py(如 `config.py`)留有未解决的 `<<<<<<< / ======= / >>>>>>>` 冲突标记 → **所有**导入它的任务 `SyntaxError: invalid syntax` 秒崩。
- 来源: `git stash pop` / merge 冲突没收尾。**gf3/uc 是 cron `reset --hard` 所以自愈; 但 gf0/cnsh 是 main 源、working tree 会留冲突**。
- 判据/防范: 提交任务前 `grep -rnE "^(<<<<<<<|>>>>>>>)" <import 到的关键文件>` 必须空; `git status` 无 `both modified`。stash pop 后务必收尾。(2026-06-04 dagger-B 实例; 冲突来自一个早已并入 HEAD 的冗余 stash, 直接 `git checkout HEAD -- config.py` + `stash drop`)。

## 12. HF_HUB_OFFLINE + `from_pretrained("hub-id")` 缓存缺失 (迁 venv 不迁 HF cache) ⚠️

- 离线节点设 `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`。若代码 `AutoTokenizer/AutoModel.from_pretrained("facebook/bart-large")` 这种 **hub repo-id** 而本地 HF cache 没有 → `LocalEntryNotFoundError` / `OSError: couldn't connect to huggingface.co` 崩。
- **跨集群迁 venv 时 HF cache (`~/.cache/huggingface`) 不会跟着走**: 老集群(uc)首跑时联网下过, 缓存在 uc home; 只把 `.venv` 搬到 cnsh vePFS → 新节点离线找不到。
- **修复**: 跳板机有代理时, **curl 小文件**直接拉到 vePFS 本地目录(hub metadata HEAD 走 huggingface_hub 常被代理挡, 但 `curl https://hf-mirror.com/<repo>/resolve/main/<file>` 能过)。X-VLA 只用 bart-large 的 **tokenizer**(config.json/vocab.json/merges.txt/tokenizer.json, 不要 1.6G 权重)。然后让路径 **env 可配**(`XVLA_BART_TOK`, 默认仍 hub-id), YAML 指到 vePFS 本地目录 + 加离线 load 预检。(2026-06-04 XVLA X3C_p0 实例, commit `638b5a5`; tokenizer 落在 `xvla/assets/bart-large-tokenizer/`)。

---

## 速查: 一个新数据集 → 提交训练的完整前置链

```
1. build 数据集 (视频目录用 feature key 命名 §3; info.total_episodes == episodes.jsonl == parquet 数 §3) → self_built/<name>/
2. compute_norm_states_fast.py --config-name <config>  (§1, 数据所在机)
3. 加 config 到 config.py + git commit && push  (§8)
4. init ckpt 在位 + size 校验完整 (§4)
5. 目标集群 git pull 到含 config + data_loader fix (§2 §8)
6. 提交 (CLI override --data.repo_id / --weight-loader.params-path 到目标集群路径 §2)
7. 验证: log 出 "Generating train split" (dataloader 过) + "Step N: loss" (训练循环)
```
