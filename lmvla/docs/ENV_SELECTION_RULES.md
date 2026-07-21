# 环境选择规则(单一事实源 · 2026-07-20 定)

> 配套勘查报告:[`ENVIRONMENTS_map_2026-07-20.md`](ENVIRONMENTS_map_2026-07-20.md)(含跨集群指纹对比与事故清单)
> **动手前先查本表。** 今日已因选错环境/硬编码路径造成多次失败提交。

---

## 1. 一张表决定用哪个解释器

| 你要做的事 | 用这个 | 备注 |
|---|---|---|
| VLA 训练 / 评测(starVLA、volc yaml) | **`$REPO/kai0/.venv`** | 两集群版本逐项一致 |
| CRAVE 特征提取 / r 场 / 分段 / LMWM 训练 | **`$REPO/kai0/.venv`**(North-E)<br>**`conda:srpo`**(gf0) | gf0 的 kai0/.venv 缺 sklearn;见 §3 待办 |
| `train_scripts/kai/data/crave_*.py` 分析脚本 | **`conda:srpo`** | 需 numpy+sklearn+matplotlib+diffusers |
| RoboTwin 仿真侧(envs / bridge) | **RoboTwin conda**(经 `ROBOTWIN_PYTHON`) | sapien/mplib/curobo 不在任何 venv |
| Qwen3-VL 特征提取 | **`$REPO/kai0/.venv`** | 需 transformers 5.x |
| **SigLIP / So400m 特征提取** | **任一**(建议 `$REPO/kai0/.venv` 统一) | ✅ 已实测**环境无关**:srpo(tf 4.57.6) vs kai0/.venv(tf 5.13.1) 逐行 cos=**1.00000000**、max\|Δ\|=3.3e-3(fp32 非确定性量级)。核查脚本 `lmwm/scripts/check_so400m_env_consistency.py`。<br>⚠️ 与 DINOv3 不同 —— 后者在 tf 4.x/5.x 间**模块嵌套深度会变**(见 §4 事故) |

**在 volc yaml 里不要手写解释器路径** —— `source train_scripts/kai/volc/_cluster_env.sh`,
之后用 `$PYTHON` / `$ROBOTWIN_PYTHON`。该脚本按挂载点自动判定集群。

---

## 2. 现存环境(2026-07-20 清理后)

| 环境 | 位置 | 用途 | 状态 |
|---|---|---|---|
| `kai0/.venv` | 两集群 | VLA 主力;North-E 上亦可跑 CRAVE 全链路 | ✅ 主力 |
| `conda:srpo` | 仅 gf0 | CRAVE/LMWM + crave_*.py 分析 | ✅ 保留 |
| RoboTwin conda | 两集群(huanqian 名下) | 仿真 | ✅ 保留,**勿动**(他人共享) |
| `kai0/.venv_py311_bak` | 仅 gf0 | 遗留 | ⚠️ **弃用**,见 §4 |
| ~~`.venv_dinov3`~~ | — | 空壳(1 个包) | 🗑️ **已删除** |
| ~~`.venv_wanvae`~~ | — | Wan-VAE 编码器已被架构定版淘汰;**且缺 numpy,其 23 个引用脚本早已跑不通** | 🗑️ **已删除**(引用已改指 srpo) |

---

## 3. 是否需要进一步合并?

**RoboTwin conda:不能合并。** 绑定 sapien 3.0.0b1 + curobo 到 torch 2.4.1+cu121,且是他人共享环境。

**`srpo` → `kai0/.venv`:有合并空间,证据如下(但未最终裁定)**

- 定版 rvalley 管线**不依赖 sklearn**(只用 scipy/numpy/pandas)
- North-E 的 `kai0/.venv` 已**端到端跑通** CRAVE 全链路(`OK_FINGERPRINT` / `OK_HF_PATH` / `OK_RVALLEY`)
- DINOv3 的 HF 与 standalone 两条代码路径经实测**比特级一致**,不构成环境约束

⇒ 只要给 gf0 的 `kai0/.venv` 补 `scikit-learn==1.7.2`(与 srpo 同版本),即可统一到 2 个环境。

**✅ 逐字段一致性已验(2026-07-20)**:
- **rvalley 分段/建对**:同一批 DINOv3 特征、149 ep,srpo(tf4.57.6)vs kai0/.venv(tf5.13.1)
  段数/边界/段脊指纹**零差异**(scipy 两侧同为 1.15.3;`find_peaks`/`gaussian_filter1d` 无版本漂移)。
  核查脚本 `lmwm/scripts/check_rvalley_env_consistency.py`。
- **So400m 编码器**:逐行 cos=1.00000000(见 §1 备注)。
- ⇒ **合并的技术障碍已清除**,只差给 gf0 补 sklearn 这一步操作(§6)。**srpo 仍保留**直到补包完成。

---

## 4. 硬性禁令

0. **⚠️ 本表未列的模型不得默认"环境无关"。** 2026-07-20 事故:`lawam` conda env(tf **5.2.0**)与
   `kai0/.venv`(tf **5.13.1**)对 DINOv3 的模块嵌套深度不同(`vision_encoder.model.layer` vs
   `.model.model.layer`)→ LAM 加载报 **204 个 key 不匹配**,本机 LIBERO eval 全线跑不了。
   `lam_model.py` 的 key 重命名已改为**按当前模型实际期望自适应**(原版硬编码"总是加一层 `.model`")。
   → **本机 VLA 评测只能用 `kai0/.venv`**;`lawam` conda env 未列入 §2,勿用。
   同理,`LMWM_DUAL_2Q=1` 等架构 env **只在环境变量、不在 config.yaml**,漏配 = `act_query` size mismatch。
1. **不得用 `py311_bak` 抽特征。** 其 transformers 4.53.2 < 4.56,`hf_dino.py` 会**静默回退**到
   `DINOv3ViTStandalone`。虽已实测两路径比特级一致(见 map §4.2),但回退是无声的,
   一旦移植版与上游脱节即产生无法察觉的差异。评测/抽取脚本应保留「必须走 HF 路径」的硬断言。
2. **不得在 entrypoint 里写集群字面量**(`/vePFS/tim`、`/vePFS-North-E`、`/home/tim`、`/vePFS/HuanQian`)。
   用 `mkyaml.py` 生成 yaml,它会**拒绝**含硬编码的 body。
3. **未纳入 git 的脚本不得用于生产产物。** 本月已出现:`p1_libero_milestone_pairs_finalarch.py` 产出的
   生产 pairs 为 5.87 段/ep,而现在无论用哪个环境复跑都是 3.44 —— 脚本被改过且**无法 diff 回原版**,
   该差异至今无法归因。
4. **产物旁应写 `_env.json`**:python/torch/transformers/sklearn/numpy 版本 **+ 脚本 git hash**。
   只记环境不够 —— 上一条的病根恰恰是缺 git hash。
5. **⛔ `lmvla/lawam` submodule 树内禁止 `git clean -fdx` / `-X`。** 2026-07-21 事故:lawam 转
   submodule 时的 `git clean -fdx`(或重新 clone)抹掉了 `latent_action_model/logs/.../lam_release/`
   那层未跟踪软链 → **所有 LaWM/LMWM eval 加载 LAM 时 FileNotFoundError**(真权重仍在 `ckpts_dl/`,
   只是软链没了)。submodule 树内裸放着 **~447G git-ignored 生产资产**(`results/` **442G**、`ckpts_dl/` 2.8G、
   `dataset/` 1.9G、`weights/` 327M),外加 `dataset/` 下**成千上万条嵌套软链**。
   - `git clean -fd`(无 `-x`)**不删** ignored,安全;**只有 `-x`/`-X` 会抹**这 447G。**cron_sync.sh 已核实无任何 clean**,
     `submodule update` 的 checkout 也不删未跟踪 → **无自动化威胁**;威胁仅来自手动误敲 `-fdx` 或再次重构。
   - 修复/自愈:`bash lmvla/lmwam/env/heal_lawam_symlinks.sh`(幂等重建 LAM release + 5 个顶层资产软链)。
     train/eval entrypoint 顶部可调它做 fail-safe。
   - **✅ 结构性根治已落地(2026-07-21,两环境):** `results/`(442G)、`ckpts_dl/`、`dataset/`、`weights/`、`logs/`
     已 `mv` 到 submodule **外部** `lmvla/lawam_local/`(超项目侧,已 gitignore),submodule 树内只留 5 条软链。
     现在 submodule 内 `git clean -fdx` **只删这 5 条软链**(heal 脚本秒重建),444G 真数据够不着。
     LAM release 软链经 `ckpts_dl` 软链**链式解引用**仍通。

---

## 5. 跨集群一致性(已实测)

| | cnsh | North-E |
|---|---|---|
| `kai0/.venv` | py3.12.13 · torch 2.6.0+cu124 · tf 5.13.1 · numpy 2.2.6 · scipy 1.15.3 | **逐项相同** |
| RoboTwin conda | py3.10.20 · torch 2.4.1+cu121 · sklearn 1.7.2 · numpy 1.26.4 | **逐项相同** |
| GPU | 8×A100-80G | 8×H20 |

⇒ **跨集群 VLA 实验可比**(唯一残留变量是 GPU 型号;若最终差距 <2pt 则不能排除硬件因素)。

---

## 6. 待办

- [x] ~~「rvalley 在 `kai0/.venv` vs `srpo` 产出是否逐字段一致」~~ → **✅ 149 ep 零差异**(§3),障碍已清除
- [ ] 给 gf0 的 `kai0/.venv` 补 `scikit-learn==1.7.2`(合并的唯一前置,纯操作)→ 补完即可退役 srpo
- [ ] 在关键脚本头部加 transformers 版本断言
- [ ] 落实 `_env.json` 产物元数据
- [x] ~~**结构性根治 submodule 内 447G 裸奔资产(见 §4.5)**~~ → **✅ 2026-07-21 完成**:
  `results/`(442G)、`ckpts_dl/`、`dataset/`、`weights/`、`logs/` 已 `mv` 到 `lmvla/lawam_local/` 并软链回,
  两环境(cnsh gf0 / cnbj North-E)各做一次,已 gitignore。软链由 `heal_lawam_symlinks.sh` 幂等重建。
