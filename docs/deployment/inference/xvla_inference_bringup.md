# X-VLA 推理 Bring-up 计划 (修订版) — 从训练好的 ckpt 到真机跑通

> **目标**: 把 fixed-pipeline 训出的 X-VLA ckpt 在 sim01 上**跑通推理 + 真机部署**。
> **关联训练侧**: [`../../training/future_plans/plans/xvla_track_x_curriculum.md`](../../training/future_plans/plans/xvla_track_x_curriculum.md) (§0 新版控制三件套 + §⚠️ 管线 bug) · [`../../../train_scripts/xvla/data/README.md`](../../../train_scripts/xvla/data/README.md) (EE6D 转换约定真相源)。
> **关联协议**: [`../multimodal_inference_protocol.md`](../multimodal_inference_protocol.md) (server-only, `action_kind="ee"` 16D)。
> **创建**: 2026-05-29 · **修订**: 2026-05-31 (③ 固件 IK 真机 execute 丝滑跑通) · **状态**: 🟢 **真机 execute 跑通, 运动丝滑无停顿 (X3.C, fp32)**。
>
> ## ⭐ 控制路径定论 (2026-05-31): EE 输出走【固件 EndPoseCtrl】, 不在主机做 IK
> 真机驱动初期严重卡顿 + 关节姿态不协调。逐项排查与修复:
> - **home 对齐 demo 起始位** (absolute-EE 对起始 OOD 敏感): IK 残差 20-40mm → ~0mm。`piper_tools/go_to_pose.py` 归到 A_0423 demo 起始位。
> - **server 确定性采样** (固定 flow-matching 噪声种子, `--seed`): 去同 obs 跨次 ~55mm 采样噪声。
> - **commanded-proprio / 开环 / RTC 连续** (node): 缓解 chunk 间不一致, 但都治标。
> - **根因 = 主机 scipy IK** (`_ee_chunk_to_joint` 60 次/chunk ~0.4s): 延迟→buffer 见底卡顿; 分支翻转→姿态乱; rate-limit 接缝→每秒踢一下。
> - **✅ 解 = ③ 固件 IK** (上游 SoftFold-Agilex 同款): node 把 world EE → base `xyz+rpy+grip` → `PosCmd` → `/pos_cmd_{l,r}` → arm_reader `EndPoseCtrl`, **固件做 IK + 笛卡尔插值**。主机零 IK → 周期 1.4s→0.33s, 无延迟/翻转/接缝踢 → **真机丝滑无停顿**。
> - **frame 已验证**: `CalFK link6`(训练编码)== 固件 `GetArmEndPose`/`EndPoseCtrl` 末端系, Δxyz=0.0mm/Δrpy=0° → EE 直送固件零偏移。
> - 开关: `ee_ctrl:=firmware`(默认) / `:=joint`(回主机 PiperDHIK, 备用)。
> - **待续**: 丝滑 ≠ 任务成功; 下一步评估 X3.C 折叠完成度 (模型质量, 与控制无关)。

---

## 0. TL;DR — 当前状态与阻塞

X-VLA 走 **server-only**: X-VLA 特有逻辑全在推理 server, 对外 emit 标准 `action_kind="ee"` 16D, 复用 `policy_inference_node --execution-mode ee_pose` 客户端。

经端到端审计 (pull `9f419d6..3c1076a` + 精读 `train_scripts/xvla/`), 真机要过必须先解决 **4 层阻塞**, 按依赖顺序:

| # | 阻塞 | 状态 | 谁来做 |
|---|---|---|---|
| **A** | 可部署 ckpt 不存在 | ✅ **X3.{A,B,C} 全拉到, 归 `ckpt_xvla/` (2026-06-01)**: `ckpt_xvla/xvla_x3c_smooth800_step_final` 等 (vis-only baseline, step30000, domain20, fixed 管线; sidecar 齐全)。所有真 X-VLA ckpt 从 `ckpt_others/` 迁入 `ckpt_xvla/` | X3.{A,B,C} 已可测 |
| **B** | server 与训练不一致 (R4) | ✅ **完成 + 实跑 (2026-05-31)**: `.venv_xvla` lerobot fork + `serve_policy_xvla.py` 加载训练同款 `XVLAPolicy`。X3.C **load missing=0/unexpected=0**, 端到端推理 (30,16) ee, quat 单位模长, 148ms/infer (fp32) | ✅ |
| **C** | 客户端 ee 消费链 | ✅ **真机 observe-only PASS (2026-05-31)**: `policy_inference_node` action_kind=ee 分支 → `PiperDHIK` (CalFK link6) 反解 16D→14D。真机实测: ee_pose ready, `/policy/actions_ee_*` ~0.85Hz, **link6 IK 失败 0-4/30 (无大面积失败, hold-last-good)**, 0 inference error, execute=false `/master/joint_*` 不发→臂不动。**input 不需 ee_pose** (★) | ✅(只差驱动) |
| **D** | server/launch/client bug (共 6) | ✅ **全修 (2026-05-31)**: ① `autonomy_launch.py` declare+转发 `execution_mode/urdf_path/calibration_yaml`; ② `start_xvla_autonomy.sh` `VENV_PY`→`.venv_xvla` + 删过时 `--image_size 224` + 删非必需 `--enable-ee-pose-input`; ③ **dtype bug**: server `config.dtype` 没跟 `.to()` 同步→`generate_actions` fp32 噪声撞权重 (已同步, 默认 fp32 对齐训练); ④ **node calib 路径**: symlink-install 下 `__file__` dirname-walk 落到 `ros2_ws/install`→calib yaml 找不到→fallback joint, 改从 `calib_dir` 父目录派生; ⑤ **node ee-attr clobber**: ee setup 写 `self._T_world_base{L,R}` 被其后 rerun-viz init 默认块重置 None (policy 自身 enable_rerun=False 不会再填)→`inv(None)` crash, 改用专属 `self._ee_T_world_base{L,R}` | ✅ |

**好消息**: 旋转 codec (`xvla_action_codec.py`) **本身是对的** (interleaved, 见 R2) — 之前错的是旧 ckpt 的训练数据, 不是 codec。

> ⚙️ **dtype = float32 (训练一致, 2026-05-31)**: 训练 `xvla_train.py:225` = `XVLAPolicy.from_pretrained(CKPT_INIT).to(device)`, **无 `.to(dtype)`/autocast/GradScaler** → 纯 fp32 训练 (`XVLAConfig.dtype` 缺省即 `"float32"`)。故 server 默认 `--dtype float32` 与官方对齐; bf16 是后续提速选项, 切换前需数值 parity 复验 (config.dtype 同步 bug 已修, bf16 现在不会再崩, 但精度未对拍)。
>
> 📊 **server 侧离线 gate 实测 (2026-05-31, X3.C, fp32, GPU3)**:
> - Step1 加载: missing=0/unexpected=0, metadata `action_kind=ee dim=16 H=30` ✅
> - Step2 合成 obs 推理 (`test_inference_server.py --check all`): 形状 (30,16) / **quat 模长 1.0000** / xyz 工作空间合理 / 敏感性 103mm / chunk 内平滑 max-jump 34mm / **延迟 148ms p95 161ms** ✅; 唯一 WARN = 同输入跨次 std 55mm (flow-matching 固有随机, 且合成 obs, 非缺陷)
> - Step3 codec/坐标 parity (离线数值): Rot6D round-trip **2.4e-12**; 200×2臂 **link6 encode→decode**: xyz **2.4e-8 m** / rot **2.8e-7** / gripper 二值化逐位吻合 ✅。server proprio 直接 `import` 训练真相源 `train_scripts/xvla/data/joint_to_ee6d.py` (全系统唯一一份) → R1/R4 by-construction 一致

> ★ **proprio 简化 (B 完成后定论)**: 训练 `observation.state` = `joint_to_ee6d_row(14D 关节)` (PiperFK link6)。server 现**直接对当前关节算 20D EE6D proprio**, 与训练同一函数 → **客户端 input 侧根本不需要 `ee_pose_left/right`** (也无需 `--enable-ee-pose-input`)。客户端只需把 server 的 **16D ee 输出**做 IK 落到关节 (路径 C output 侧)。obs 只要标准 `images{top_head,hand_right,hand_left}` + `state(14)` + `prompt`。
>
> 同时修正 R4 措辞: 训练 **绕过 processor** 直接 `model.forward(dataset_batch)`, 故图像是 **`/255` 的 [0,1], 不做 ImageNet 归一** (`processor_xvla` 的 ImageNetNormalize 步骤在该管线**未被调用**); server 已按 `multi_domain_dataset.py` 精确镜像 (resize_pad 256/256/224 → CHW → /255)。
>
> ⚠️ **2026-06-04 更正 (本段已 stale)**: 上述"不做 ImageNet 归一"**仅对 P0 之前的旧 ckpt 成立**。**P0 修复 (2026-06-01) 起训练/部署都做 ImageNet 归一** (`multi_domain_dataset.py:177` + `serve_policy_xvla.py:258`), 且这与上游官方一致 (上游训练 `dataset.py:68` + 部署 processor **本来就做**归一; 旧管线缺失正是真机抖动根因之一)。详见 [`xvla_upstream_vs_local_consistency.md`](xvla_upstream_vs_local_consistency.md) §1/§4。另: 图像最终经 lerobot `_prepare_images` 统一 letterbox 到 **224** (非 256, `resize_imgs_with_padding=[224,224]`) 才喂模型。

---

## 1. 架构 (server-only; server 内部改用 lerobot 训练类)

```
┌─ 推理 server (port 8003) ──── X-VLA 特有逻辑全在此 ────────────────────────────┐
│  R4方案②: 用训练同款 lerobot.policies.xvla XVLAPolicy 加载 (非 upstream repo)   │
│  obs in (标准 multimodal 协议 §B.1.3):                                          │
│    images.{top_head,hand_left,hand_right} uint8  (训练预处理: resize-only 224, │
│                                                    无 ImageNet 归一 — 见 R4)   │
│    state(14) — gripper ch [6],[13] (注: 模型 preprocess 把 proprio gripper 清零)│
│    ee_pose_left/right(7) world xyz+quat_wxyz   ← 客户端 --enable-ee-pose-input  │
│         ↓ 7D world EE(link6) → 20D arm-base EE6D (inv(T_world_base)+interleaved)│
│    XVLAPolicy 生成 (flow-matching ODE) → (H=30, 20D, abs xyz)                  │
│         ↓ 20D arm-base EE6D → 16D world [xyz,quat_wxyz,grip]×2                  │
│  out: actions (30,16), action_kind="ee"                                        │
└────────────────────────────────────────────────────────────────────────────────┘
         ↓ msgpack WebSocket (标准协议)
┌─ policy_inference_node --execution-mode ee_pose --enable-ee-pose-input ────────┐
│  obs: link6 FK(joint)+T_world_base → ee_pose (R1: link6, 不是 gripper_base!)    │
│  收 16D ee chunk → world→base→ link6-target IK → /master/joint_{l,r}            │
│  [整段缺失, 待建 — 见 §4 C]                                                      │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ckpt 状态 — 旧作废, 新待产

### 2.1 ❌ 旧 x3a/b/c_stage_a (`ckpt_xvla/xvla_x3{a,b,c}_stage_a_*`) — **全部作废, 禁止部署**
训练管线有 3 个确定 bug (见 curriculum §⚠️ + `train_scripts/xvla/data/README.md`):
1. **Rot6D block 排布** (`R[:,:2].T.flatten()` = `[r00,r10,r20,r01,r11,r21]`) ≠ 上游 interleaved → 部署 codec 解码会 garble 旋转。
2. **gripper 未二值化** (灌原始米值) → BCE 通道学不会闭合。
3. **decode_frame `frame.index`** 失效 → vis/kai parquet 域**可能全黑图**。

→ 旧 ckpt + 当前(正确)codec = 姿态错乱 + gripper 不闭 + (可能)没学到视觉。**不可上真机。**

### 2.2 ⏳ 新 fixed-pipeline 控制三件套 (训练中, uc, ETA ~6h) — **部署目标**
统一 vis 数据 `A_0423_0527`、统一超参 (30k/lr5e-5/warmup500/freeze1000)、fixed 管线 (interleaved rot6d + 二值 gripper + decode 修复)。

| 实验 | 域组成 | uc output_dir | 部署候选 |
|---|---|---|---|
| **X3.C** (vis-only) | A_0423_0527 | `xvla_A_0423_0527` (uc01) | baseline 候选 |
| **X3.B** (+kai) | kai + A_0423_0527×7 | `xvla_x3b_a0423` (uc02) | ⭐ 主候选 |
| **X3.A** (+kai+xvla) | +xvla_soft_fold×2 | `xvla_x3a_a0423` (uc03) | 对照 |

- ckpt 格式: `state_dict.pt` (`{"model_state","step"}`, key 前缀 `model.`) + sidecar (待确认是否新版生成)。
- **部署目标待干净 eval 定**: 旧 "X3.B 完胜" 结论作废需重验; 真机首跑建议主候选 **X3.B_a0423** (或 X3.C baseline)。
- 推理 force `domain_id=20` (vis)。
- **动作**: 训完 → 拉到 sim01 `ckpt_others/` (走 TOS 中转, 拉完即 `tosutil rm`)。

---

## 3. ⭐ 正确性契约 (R1–R4) — 真机不出错的核心, 全部已用真相源定死

> 真相源: `train_scripts/xvla/data/joint_to_ee6d.py` (kai/vis 转换) + `data/README.md` (官方一致性 gate) + upstream `xvla/X-VLA` codec/action_hub。

### R1 — EE link = **link6** (不是 gripper_base)
训练 `joint_to_ee6d.py`: `C_PiperForwardKinematics(0x01)` 的 `CalFK`, 取 `result[-1]` = **link6** (DH FK, 2° j2/j3 offset), xyz mm→m, rpy(deg)→matrix。
**部署硬约束**:
- obs `ee_pose_*` 必须用 **PiperFK link6** (`calib/piper_fk.py`, `fk_homogeneous`=link6), **不能用 `calib/piper_ik.py`** (那是 gripper_base, 差 +0.1358m)。
- action 的 IK 必须解到 **link6**。`piper_ik.py` 解 gripper_base → 需对 target 施加 −0.1358m(link6 沿 link6 局部 +z 的反向偏移)再解, 或换 link6-target IK。**否则全程 13.58cm 系统偏差**。

### R2 — Rot6D = **interleaved `[r00,r01,r10,r11,r20,r21]`** (codec 已对)
fixed 转换 `rot6d = R[:, :2].flatten()` = `[r00,r01,r10,r11,r20,r21]`。
- 部署 codec `xvla_action_codec.rotation_matrix_to_interleaved_6d` / `interleaved_6d_to_rotation_matrix` **逐行匹配** (= 上游 `quat_to_rotate6d` = SoftFold `rotation.py`)。✅ **codec 不用改**。
- ⚠️ 仅对 **新 fixed ckpt** 成立; 旧 ckpt 是 block 排布, 不兼容 (§2.1)。

### R3 — gripper = **二值化** {0,1}
训练 `out[...9] = (gripper_m*50 < 1.0)` → <0.02m 记 1(闭); action_hub `EE6DActionSpace` 对 ch(9,19) 用 BCE + postprocess sigmoid。
- server `_ee6d_to_world_quat_grip` `binarize_gripper=True`: sig>0.5→close_value 否则 open_value。方向一致 ✓。
- **待核实**: server 默认 `open=0.0656m / close=-0.0055m` 是 SoftFold 值; vis 机器人 Piper 夹爪实际行程是否一致, 不一致用 `--gripper_open_value/--gripper_close_value` 覆盖 (负的 close 可能被硬件 clip)。
- 注: proprio gripper 被模型 preprocess 清零, 故 obs gripper 归一化无关紧要。

### R4 — server 预处理/模型类必须与训练一致 → **方案② 装 lerobot xvla** ✅ 完成
训练实际用 **`lerobot.policies.xvla.modeling_xvla.XVLAPolicy`** (非 upstream repo), 且**绕过 processor** 直接 `model.forward(dataset_batch)` → 以 `multi_domain_dataset.py` 为准: **图像 = resize_pad 256/256/224 → CHW → `/255` ∈[0,1], 不做 ImageNet 归一** (`processor_xvla` 的 ImageNetNormalize 步骤**未被调用**); proprio = `joint_to_ee6d_row(14D)`; language = BART max50; abs xyz; chunk=30。
- **已落实 (2026-05-29)**: `.venv_xvla` 装 lerobot 0.4.4 fork (sim01_deployment §3.6); `serve_policy_xvla.py` 重写为 `PreTrainedConfig.from_pretrained(base)`+`XVLAPolicy`+ 灌 `state_dict.pt["model_state"]`, 镜像上述 batch。旧 x3b 联调: **missing=0/unexpected=0**, `predict_action_chunk`→(1,30,20)→16D world, 输出 (30,16)/quat 单位/finite。
- 20D→16D 转换 (codec + T_world_base) 留在 server。

---

## 4. 缺口与待建 (B/C/D)

### B. server 改用 lerobot XVLAPolicy (R4 方案②) ✅ 完成
`kai0/scripts/serve_policy_xvla.py` 已重写: 加载训练同款 `XVLAPolicy` + 镜像训练预处理 (图像 /255 无归一、proprio=joint_to_ee6d、BART token、domain20) + 20D→16D world 输出。依赖在独立 `.venv_xvla` (cu128, 不碰旧 venv)。新增资产 `kai0/assets/xvla/lerobot_base/config.json` (来自 base ckpt, 建 XVLAConfig 用)。

### C. 客户端 ee 消费链 (整段缺失, 必建) — **大头, 待写**
`ros2_ws/src/piper/scripts/policy_inference_node.py` 实测 `action_kind/PiperIK/IK-chunk/EE publishers/execution_mode` **全 0 次** (git 全历史无, 从未落地)。需建 (对齐 `multimodal_inference_protocol.md` §B, **EE 用 link6 见 R1**):
- ★ **input 侧不需 ee_pose**: server proprio 走当前关节 (R4/§0★) → **obs 无需 `ee_pose_left/right`, 也不用 `--enable-ee-pose-input`**。`_get_observation` 不改 (维持标准 images+state+prompt)。
- `infer` 结果按 `action_kind=="ee"` 分支 (server metadata 已声明 ee/16/30): **16D world → world→base (inv T_world_base) → T_base_link6 → IK → [H,14] joint**。
  - **IK 必须用训练同一 DH 模型 (CalFK link6)** → 新 `calib/piper_dh_ik.py::PiperDHIK` (scipy least_squares around `C_PiperForwardKinematics(0x01)`)。⚠️ **不能用 `calib/piper_ik.py` (ikpy/URDF)**: 实测 ikpy 解经 DH-FK 回算 link6 偏差达 **~5.7cm / 6.7°** (URDF≠DH), 真机会系统性偏掉; PiperDHIK round-trip 实测 **0.0009°/0.36µm**。
- `_ee_chunk_to_joint` (seed=上一步解链, 起始 seed=当前关节; IK `ok=False`(超 tol) 时 hold last good + 计数告警)。
- 新 publishers `/policy/actions_ee_{left,right}` (PoseStamped, world) + `/policy/actions_gripper_{left,right}` (Float32) — 监控用, 直接发 server 的 16D 首帧。
- 新参数 `execution_mode` (joint|ee_pose), `urdf_path`, `calibration_yaml`; `enable_ee_pose_input/enable_depth_input` 对 XVLA **不需** (input 走关节算 proprio)。
- StreamActionBuffer / 平滑 / RTC / jump-protect / publish 仍在 **joint 域** (IK 后, 注入点在 `actions=result['actions']` 之后立即转 joint), 与现有 joint 路径共用 → **旧路径 byte-identical**。
- **离线验证 (2026-05-29, 无真机)**: PiperDHIK self-test PASS; server-encode→node-decode→IK 全链路往返还原 **0.0001°**; node `py_compile` OK。

### D. 2 个小 bug (低风险, 可先修)
- `ros2_ws/src/piper/launch/autonomy_launch.py`: 声明 + 转发 `execution_mode/enable_ee_pose_input/enable_depth_input` (+ calibration_yaml/urdf) 到 node (现 0 次)。`start_autonomy.sh` 已会传 `execution_mode:=` 等, 但 launch 不接。
- ~~`start_scripts/xvla/start_xvla_autonomy.sh`: client 模式调 `./start_scripts/start_autonomy.sh`, 但 reorg 后实际在 `start_scripts/kai/start_autonomy.sh` → 修路径。~~ **已修 (2026-05-29)**: 脚本移到 `start_scripts/xvla/start_xvla_autonomy.sh`, client 路径改为 `./start_scripts/kai/start_autonomy.sh`, `REPO_ROOT` 深度同步修正。

---

## 5. Bring-up 步骤 (修订版, 依赖顺序 + gate)

### Step 0 — 等新 ckpt + 拉取 🔴 阻塞
新三件套训完 → 拉到 `ckpt_others/`。确认每个含 `state_dict.pt` + (新版) sidecar; 干净 eval 选定部署目标 (主候选 X3.B_a0423)。

### Step 1 — server 改造 (R4 方案②) + 单机自检
装 lerobot xvla → 重写 serve 加载 XVLAPolicy → 起 server (:8003)。
**Gate**: 加载无 missing/unexpected; metadata `action_kind=ee, action_dim=16, H=30`; 图像预处理 = resize-only 224 (无 ImageNet 归一, 与训练核对一致)。

### Step 2 — 合成 obs 打一次推理 (协议测试工具已就绪)
```bash
python3 start_scripts/kai/test_inference_server.py --check all --port 8003
```
metadata 驱动: 自动识别 ee + 补 ee_pose + 16D 校验。
**Gate**: shape `(30,16)`; 逐臂 xyz 工作空间内; **quat 单位模长**; grip ∈ bracket; 敏感性 PASS; `infer_ms` 基线。

### Step 3 — codec / 坐标 parity (offline, 数值) 🔴 关键正确性
1. Rot6D 往返: `python kai0/scripts/xvla_action_codec.py` (~1e-7) ✓ 已自带。
2. **link6 round-trip**: 14D joint → PiperFK **link6** world EE → `_ee_world_to_base_6d` → `_ee6d_to_world_quat_grip` → 还原同一 world pose (<1e-5)。**注意全程 link6, 与训练 R1 一致**。
3. proprio parity: server 构造的 20D state20 与训练 `joint_to_ee6d_row` 输出对同一 joint 应一致 (可直接 import `train_scripts/xvla/data/joint_to_ee6d.py` 对拍)。

### Step 4 — gripper 映射核对 (R3)
核 vis 机器人 open/close 米值, 必要时 CLI 覆盖。

### Step 5 — 客户端 ee 路径 (建 C+D) + observe-only 对接 🟡 上真机不动臂
```bash
./start_scripts/xvla/start_xvla_autonomy.sh server <新ckpt>
./start_scripts/xvla/start_xvla_autonomy.sh client        # observe-only
```
**Gate**: `/metadata` 读到 ee/16/30; `/policy/actions`(14D IK解) + `/policy/actions_ee_*` 发布; **link6 IK 无大面积失败**; Rerun EE 轨迹连续合理; execute=false 臂不动。

### Step 6 — 真机驱动 🟢
确认轨迹合理后 `ros2 topic pub /policy/execute std_msgs/Bool 'data: true' --once`。
**Gate**: 臂平滑跟随无大震荡 (参考 `realtime_vla/ee_stability_layer1.md` §3.4/3.5); 完成 flatten/fold 合理动作。

---

## 6. 风险表

| 风险 | 说明 | 缓解 |
|---|---|---|
| **R1 link6 vs gripper_base** | 训练 link6, 若部署用 piper_ik(gripper_base) → 13.58cm 偏差 | obs 用 PiperFK link6; IK 解 link6 (target 减 0.1358m 或换 link6 IK); Step 3.2 round-trip 校 |
| **R4 模型类/预处理** | 训练 lerobot port (无归一), 误用 upstream+ImageNet → 推理退化 | 方案②: 装 lerobot xvla 加载训练类; Step1 核对图像管线 |
| **旧 ckpt 误用** | x3a/b/c 作废, 易误拿 | 只部署新 fixed 三件套; sidecar 标注 |
| **lerobot fork 不可得/版本漂移** | sim01 `.venv_5090` 无 policies.xvla | 从 uc `X-VLA-env` 锁定 fork+版本装入; state_dict key 映射验证 |
| **gripper 行程** | SoftFold 默认 ≠ vis 机器人 | Step4 核 + CLI 覆盖 |
| **推理时延** | 0.9B+Florence2+ODE | Step2 记基线; 必要时降 steps / 配 RTC |
| **未 commit** | server/client/脚本改动 | 真机验证后再提交 |

---

## 7. 文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `kai0/scripts/serve_policy_xvla.py` | 未跟踪, **待重写(B)** | 改用 lerobot XVLAPolicy 加载 + 训练预处理 |
| `kai0/scripts/xvla_action_codec.py` | 未跟踪, **不改** | interleaved Rot6D, 已对(R2) |
| `kai0/assets/xvla/processor/` | 未跟踪 | upstream Florence2 processor — 方案②下不再用其归一 |
| `train_scripts/xvla/data/joint_to_ee6d.py` | 已提交 | **R1/R2/R3 真相源**; Step3 对拍用 |
| `calib/piper_dh_ik.py` | **本次新增** | **DH-model IK (CalFK link6)** — path C 用它反解; ikpy/URDF 不可用(差 5.7cm) |
| `calib/piper_fk.py` | 已提交 | DH link6 FK (rerun viz); PiperDHIK 内部用 CalFK 同源 |
| `calib/piper_ik.py` | 已提交 | gripper_base ikpy — **XVLA 路径不用**(URDF≠DH); 留作其他用途 |
| `ros2_ws/.../policy_inference_node.py` | 已改(未提交), **C 完成** | action_kind=ee 分支 + PiperDHIK + EE publishers + params; 旧 joint 路径 byte-identical |
| `ros2_ws/.../launch/autonomy_launch.py` | 已提交, **待修(D)** | 转发新参数 |
| `start_scripts/xvla/start_xvla_autonomy.sh` | 未跟踪, client 路径 **已修(2026-05-29)**; `VENV_PY` 待改 | 移到 xvla/, client 路径→kai/ + REPO_ROOT 修正; `VENV_PY` 仍待指 `.venv_xvla` (server 改造时) |
| `start_scripts/kai/start_autonomy.sh` | 已改(未提交) | 已解析+转发新 flag ✓ |
| `start_scripts/kai/test_inference_server.py` | 已扩(未提交) | 协议驱动测试, Step2 用 ✓ |
| 新 ckpt `ckpt_xvla/xvla_x3{a,b,c}_{a0423,smooth800}` | ✅ 已拉 (归 ckpt_xvla/) | 部署目标 |

---

## 8. 决策点
- **D0**: 新三件套干净 eval → 选部署 ckpt (主候选 X3.B_a0423)。
- **D1 (Step3)**: link6 round-trip + proprio 对拍通过 = 坐标/codec 正确。不过 → 查 R1 link6 偏移 / R2 排布。
- **D2 (Step6)**: 真机表现 vs pi05 vis baseline (同硬件), 决定 Track X 是否进真机主线。
