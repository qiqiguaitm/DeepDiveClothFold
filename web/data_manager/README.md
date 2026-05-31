# data_manager — 双臂 VLA 数据采集 UI

按 `../ui_data_collection_plan.md` 实现的前后端骨架，已接真 ROS2 + 真 mp4/parquet 落盘。

## 结构
```
data_manager/
├── backend/        FastAPI + 统计服务 + 录制状态机（PyAV AV1 + pyarrow）+ 模板
├── frontend/       React + TypeScript + Vite
├── config/         collection_templates.yml
├── data_mock/      旧版本地开发占位 (新数据写到 /data1/DATA_IMP/KAI0, 见下)
└── run.sh          一键启动 CAN + 机械臂 + 相机 + 后端 + 前端
```

## 一键启动
```bash
./run.sh start        # 启动全部
./run.sh status       # 查看各服务状态
./run.sh logs backend # 跟踪单个服务日志
./run.sh stop         # 停止全部

# 跳过部分模块（例如已外部启动或仅调前端）
SKIP_CAN=1 SKIP_ARMS=1 SKIP_CAMERAS=1 ./run.sh start
```
启动后：
- 前端 http://HOST:5173/
- 后端 http://HOST:8787/  (REST docs: `/docs`；WS: `/ws/status`)

## 后端 venv 构建

### 为什么不能直接 `python3 -m venv`
后端需要 `import rclpy`。ROS2 Jazzy 只分发了 **Python 3.12** 的 rclpy C 扩展
(`/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/_rclpy_pybind11.cpython-312-*.so`),
其它 Python 版本导入时会报
`No module named 'rclpy._rclpy_pybind11'` → 后端静默回落到 `MockBridge`, UI 上
CAN / cameras / teleop 全红。因此 venv 里的 `python` 必须是 `3.12`。

sim01 的复杂性:
1. `python3` 指向 miniconda 的 3.13 (不匹配 rclpy);
2. `/usr/bin/python3.12` 存在, 但系统没装 `python3.12-venv`, 所以 `python3.12 -m venv`
   会报 `ensurepip is not available`;
3. 我们不总是有 `sudo apt install python3.12-venv` 的权限.

下面给出两条路径: 有 sudo 走 A (干净), 没 sudo 走 B (workaround, 与 gzllll 的现有 venv 一致).

### A. 有 sudo —— 直接用系统 3.12
```bash
sudo apt install -y python3.12-venv

cd web/data_manager/backend
rm -rf .venv
/usr/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

### B. 无 sudo —— "3.13 骨架 + 3.12 解释器" 组合
用 miniconda 3.13 (有 `venv` 模块) 生成 venv 骨架, 然后把 `bin/python` 换成
`/usr/bin/python3.12`, 再用 `get-pip.py` 给 3.12 装 pip (因为没 `ensurepip`):
```bash
cd web/data_manager/backend
rm -rf .venv

# 1) 用 miniconda 的 3.13 建骨架 (activate / site-packages / include)
/data1/miniconda3/bin/python3 -m venv .venv

# 2) 把 python* 符号链接全部指向系统 3.12, 让实际运行的解释器是 3.12
rm .venv/bin/python .venv/bin/python3 .venv/bin/python3.13
ln -sf /usr/bin/python3.12 .venv/bin/python
ln -sf python .venv/bin/python3
ln -sf python .venv/bin/python3.12
ln -sf python .venv/bin/python3.13

# 3) 没 ensurepip, 用官方 bootstrap 给 3.12 装 pip
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py

# 4) 装依赖 (会写入 lib/python3.12/site-packages)
.venv/bin/pip install -r requirements.txt
```
产出的 venv: `pyvenv.cfg` 里 version 仍然是 3.13 (骨架来源), 但 `bin/python -V`
返回 `Python 3.12.x`, 且 `lib/python3.12/site-packages` 有全套包。

### 验证
```bash
source /opt/ros/jazzy/setup.bash
source $REPO_ROOT/ros2_ws/install/setup.bash
.venv/bin/python -c "import rclpy; print(rclpy.__file__)"
# 期望: /opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py

./run.sh start
grep RclpyBridge logs/backend.log
# 期望: [ros_bridge] RclpyBridge online    (不是 "using MockBridge")
```
UI 上 `CAN 左/右`, `teleop`, 三路相机 fps 应全部变绿。

### 常见环境变量
```bash
export KAI0_TEMPLATES=../config/collection_templates.yml
# ROS bridge: auto(默认) / mock(强制假数据, 不依赖 rclpy 可用)
# export KAI0_ROS_BRIDGE=mock
# 数据根: 默认 /data1/DATA_IMP/KAI0 (与 repo 隔离, git clean / 删 venv 不会误清)
# export KAI0_DATA_ROOT=/some/other/path
```
手动启动后端 (`run.sh` 以外):
```bash
cd backend
source /opt/ros/jazzy/setup.bash && source $REPO_ROOT/ros2_ws/install/setup.bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8787 --reload
```

## 前端（手动）
```bash
cd frontend
npm install && npm run dev   # http://localhost:5173
```
Vite 代理 `/api`、`/ws` → `localhost:8787`。

## 角色切换
顶栏按钮切换 collector/admin，前端写入 `localStorage.role`，请求带 `X-Role`；
管理员路由在后端 `require_admin` 校验。

## ROS2 桥 (`backend/app/ros_bridge.py`)
后端常驻 `rclpy` 订阅：
- 关节：`/master/joint_{left,right}` + `/puppet/joint_{left,right}`（见 `config/pipers.yml`）
- 相机：`ros2_topic_color` 三路 RealSense（见 `config/cameras.yml`），CameraInfo 算 fps/latency，Image 同时缓存 JPEG（供 MJPEG 端点）和原始 RGB ndarray（供录制）
- 暴露 `get_health / get_joint_state / get_camera_health / get_latest_jpeg / get_frame_rgb / get_state_action`

当 `rclpy` 或 YAML 不可用时自动退回 `MockBridge`（返回正弦关节值 + 渐变条纹图），端到端链路仍可跑通。强制 mock：`KAI0_ROS_BRIDGE=mock`。

## 录制落盘
`backend/app/recorder.py` — `start() → RECORDING → save()/discard()`：
- `start()` 起 30Hz 采集线程，从 bridge 拉三路 RGB 帧 + 14 维 state/action
- PyAV 编码：默认 `libx264`（双击/浏览器都能播）；`KAI0_VIDEO_CODEC=av1` 切换为 `libsvtav1`/`libaom-av1` 匹配 LeRobot 归档格式；`480×640`、`yuv420p`、30fps
- `save()` flush 容器，`pyarrow` 写 LeRobot v2.1 parquet，更新 `meta/{episodes.jsonl, tasks.jsonl, info.json}`；stats 服务自动 upsert
- `discard()` 关闭容器并删除半成品文件

产出目录遵循 LeRobot v2.1：
```
<KAI0_DATA_ROOT>/Task_A/<base|dagger>/
├── data/chunk-000/episode_NNNNNN.parquet    # obs.state[14], action[14], timestamp, frame_index, ...
├── videos/chunk-000/{top_head,hand_left,hand_right}/episode_NNNNNN.mp4  # AV1
└── meta/{episodes.jsonl, tasks.jsonl, info.json}
```

state = 从臂 (puppet) 关节 = 机器人真实位姿；action = 主臂 (master) 关节 = 遥操指令；
顺序 `[L_j1..L_j6, L_gripper, R_j1..R_j6, R_gripper]`。

### 视频播放
默认录 H.264，本地播放器/浏览器双击直接播。若设 `KAI0_VIDEO_CODEC=av1` 录 AV1（匹配
LeRobot 归档规范），视频端点会在线用 ffmpeg 转成 H.264 碎片 mp4 给浏览器；`?raw=1` 拿原始文件。
已有 AV1 文件想一次性转 H.264：`bash backend/tools/transcode_av1_to_h264.sh [DATA_ROOT]`，
原文件备份到 `*.mp4.av1.bak`，同步把各 `meta/info.json` 里 `video.codec` 改成 `h264`。

## 右下角健康提示 (FloatingHealth)
`frontend/src/components/StatusBar.tsx` 内 `collectFailures` 汇总右下角大牌判据：
- `health.ros2 / can_left / can_right / teleop` 任一为 false
- 任一期望相机 (`top_head / hand_left / hand_right`) 缺失，或实测 `fps < 25`
- 录制状态为 `ERROR`
- 后端下发的 `warnings`（如 `low_disk:<free>GB`）

相机 `dropped` 计数仅用于 `CameraGrid` 展示，**不参与异常判定**，避免瞬时抖动刷红。

## 目录布局 (v2: task/subset/date)

数据布局演变:
- v0 (扁平)   `<DATA_ROOT>/Task_X_YYYY-MM-DD/<subset>/...`
- v1 (按日期) `<DATA_ROOT>/Task_X/YYYY-MM-DD/<subset>/...`
- v2 (按子集) `<DATA_ROOT>/Task_X/<subset>/YYYY-MM-DD/...`   ← **当前**

```
<DATA_ROOT>/
├── Task_A/
│   ├── base/
│   │   ├── 2026-04-16/        # (data/, videos/, meta/ ...)
│   │   ├── 2026-04-17/
│   │   └── 2026-04-25/
│   └── dagger/
│       └── 2026-04-16/
├── Task_E/
│   └── base/
│       ├── 2026-04-17/
│       └── 2026-04-20/
└── Task_P/...
```

v2 把 subset 抬到日期之上 — 同一种数据 (`base` / `dagger`) 跨日期连成一棵子树,
方便整 subset 做训练 / 同步 / 备份。

内存 / SQLite / API URL / UI 里 `task_id` 仍用 **compound** 形式 (`Task_A_2026-04-16`),
只有磁盘路径变了。所有构造/解析路径都走 `backend/app/layout.py`, 新写一律 v2,
读旧数据按 v2 → v1 → v0 优先级回退, 支持"迁了一半"的过渡状态。

### 迁移历史数据
```bash
# dry-run 看清楚要动什么 (同时识别 v0 → v2 和 v1 → v2)
./backend/.venv/bin/python backend/tools/migrate_layout.py
# 确认无误后真干
./backend/.venv/bin/python backend/tools/migrate_layout.py --apply
```
脚本只会动数据子集 (内部带 `data/` / `meta/` / `videos/` 之一), 其他东西
(`.tar` 备份 / `ckpt_downloads/` / `*.py`) 原地不动。同盘 mv 零拷贝, 秒级完成。
迁移幂等: 已经是 v2 的目录会被跳过, 远端 (gf vePFS / bja2) 也用同一逻辑就地 rename
(参见 `tools/kai0_migrate_v2.sh`, 已分发到远端 `/tmp/`), 无需重传 165 GB 历史数据。

## 实时数据同步 gf (vePFS/gpfs 共享卷)

`recorder.save()` 成功后会异步 rsync 刚写完那条 episode 所在的 `<task>/<date>/<subset>/`
目录到 **gf0** 的 `/vePFS/visrobot01/KAI0/`。因为 gf0 和 gf1 两端的 `/vePFS` 挂的是同一块
gpfs 卷 (`fs_vepfs-cnsh075262e1f815`), 推 gf0 = 到达 gf1, 不必重复推。
不阻塞 UI, 失败只 log, **不加 `--delete`**, 本地误删不会传到云端。

### 一次性准备 (需要 root)
`/vePFS/visrobot01/` 默认是 `root:root 755`, `tim` 无法写入。在 gf 任意一台:
```bash
ssh -p 55555 tim@14.103.44.161 'sudo chown -R tim:tim /vePFS/visrobot01/KAI0'
```
之后整个 `KAI0/` 子树都归 tim, 不再需要 sudo。

### 常用配置
```bash
# 关同步
KAI0_SYNC_ENABLED=0 bash start_scripts/start_data_collect.sh

# 换推 gf1 (两端都连同一块 gpfs, 二选一即可)
export KAI0_SYNC_REMOTES='[{"name":"gf1-vepfs","user":"tim","host":"14.103.44.161","port":11111,"dest_root":"/vePFS/visrobot01/KAI0"}]'

# 调优
KAI0_SYNC_RETRIES=3         # rsync 失败重试次数
KAI0_SYNC_BACKOFF_S=2       # 指数退避基值
KAI0_SYNC_TIMEOUT_S=600     # 单次 rsync 上限
```

### 监控
```bash
curl -s http://127.0.0.1:8787/api/sync/status | jq
tail -f web/data_manager/logs/sync.log

# 管理员一次性把本地所有 subset 全推一遍 (迁移后首次对齐用)
curl -X POST -H 'X-Role: admin' 'http://127.0.0.1:8787/api/sync/all'
curl -X POST -H 'X-Role: admin' 'http://127.0.0.1:8787/api/sync/all?only_task=Task_A'
```

前置条件: ssh key 已配好; `/vePFS/visrobot01/KAI0/` 对 tim 可写 (见上); rsync 3.2.3+
(有 `--mkpath` 自动建目标父目录)。

## USB 踏板启停 (pedal service)
`run.sh` 启动时会顺带拉起 `backend/tools/pedal_listener.py`, 把踏板键按下变成
`POST /api/recorder/toggle` 调用, 和鼠标"开始/保存"按钮互斥共用后端状态机:
- **IDLE → 启动**: 跑和前端一致的 preflight (ROS2/CAN/teleop/3 路相机 fps/recorder 错误/warnings),
  任何一项失败都**不会**启动, 409 响应里带 `failures` 列表; 通过则用当前 session 里记到的
  `template_id` + `operator` 启动
  - Session 自动同步: 前端在用户改 task/prompt 下拉或敲操作员姓名时, 300ms 去抖后
    PUT `/api/session` 推给后端 (App.tsx); 所以只要 UI 里选了 template + 填了姓名,
    踏板立刻可用, 不需要先点一次"开始"
  - 无头冷启动备选: 没 UI 的场景可 export `KAI0_DEFAULT_TEMPLATE` + `KAI0_DEFAULT_OPERATOR`,
    后端启动时读入作为初始 session; 或直接 `curl -X PUT .../api/session -d '{...}'`
- **RECORDING → 保存**: `success=True, note=pedal, scene_tags=[]` 一键完结; 想带
  结果/场景标签/备注还是用鼠标按钮
- **SAVING / ERROR**: 一律拒绝
- 互斥由后端 `Recorder._lock` 保证, 鼠标踏板连按、两人同时按都不会 double-start

设备识别按 **VID:PID** 做, 不绑 `/dev/input/eventN`, 换 USB 口无需改配置;
默认匹配我们手上的踏板 (STM32 `0483:5750`, 映射 F3). 其他型号改:
```bash
SKIP_PEDAL=1                           # 不启用
PEDAL_VID=0483 PEDAL_PID=5750          # 换踏板时用 lsusb 查 VID:PID
PEDAL_KEY=KEY_F3                       # 或 KEY_F4 / KEY_A ... (见 evdev.ecodes)
PEDAL_EDGE=release                     # release(松开触发, 默认) | press(踩下)
PEDAL_DEBOUNCE_MS=500                  # 两次 toggle 最小间隔
```

热插拔: listener 启动时 / 运行中掉线时会回到 "按 VID:PID 扫描 → 2s 退避重试"
循环; 用 `evdev.grab()` 独占, 防止按键串入终端/浏览器触发 F3 快捷键.

**权限 (一次性设置)** — `/dev/input/event*` 默认 `crw-rw----  root:input`, 非 root 必须
在 `input` 组. 两条路二选一:

1. **加用户进 input 组** (推荐, 一劳永逸):
   ```bash
   sudo gpasswd -a "$USER" input && newgrp input  # 或直接重新登录
   ```

2. **udev 规则** (只对这只踏板放权, 最小权限):
   ```bash
   sudo cp config/99-kai0-pedal.rules /etc/udev/rules.d/
   sudo udevadm control --reload && sudo udevadm trigger
   # 如需换不同 VID:PID 的踏板, 先编辑规则里的 idVendor/idProduct
   ```

验证:
```bash
./run.sh status      # pedal 一行应该 running
./run.sh logs pedal  # 应该看到 "grabbed /dev/input/event... (HID 0483:5750)"
# 踩一下踏板, 日志出现: "pedal fired ... toggle 200 in ... ms: ..."
```

若 `pedal failed to start`, 查 `logs/pedal.log`:
- `PermissionError: [Errno 13] Permission denied: '/dev/input/eventN'` → 权限没配
- `waiting for pedal VID:PID=0483:5750 ...` 一直不变 → 踏板没插 / VID:PID 对不上,
  先跑 `lsusb | grep -i <型号关键字>` 确认

## 环境变量速查
| 变量 | 作用 | 默认 |
|------|------|------|
| `KAI0_DATA_ROOT` | 采集落盘根目录 (与 repo 隔离避免误删) | `/data1/DATA_IMP/KAI0` |
| `KAI0_TEMPLATES` | 采集模板 yml | `<repo>/web/data_manager/config/collection_templates.yml` |
| `KAI0_PIPERS_YML` | 机械臂配置 | `<repo>/config/pipers.yml` |
| `KAI0_CAMERAS_YML` | 相机配置 | `<repo>/config/cameras.yml` |
| `KAI0_ROS_BRIDGE` | `auto` / `mock` | `auto` |
| `KAI0_JPEG_QUALITY` | MJPEG JPEG 质量 | `60` |
| `KAI0_JPEG_STRIDE` | MJPEG 下采样 | `2` |
| `KAI0_DEFAULT_TEMPLATE` | 踏板启动时的默认 template_id (IDLE 且无记忆时使用) | — |
| `KAI0_DEFAULT_OPERATOR` | 踏板启动时的默认 operator (同上) | — |
| `SKIP_PEDAL` | `1` 跳过踏板监听 | `0` |
| `PEDAL_VID` / `PEDAL_PID` / `PEDAL_KEY` / `PEDAL_EDGE` / `PEDAL_DEBOUNCE_MS` | 踏板参数覆盖 | 见上 |
| `BACKEND_URL` | 踏板调 toggle 的后端 URL | `http://127.0.0.1:8787` |
