# 按机器人区分的 CAN 激活教程

本文说明 `piper_tools/activate_can.sh` 的使用方式，以及为什么需要按机器人名字选择不同的 SocketCAN 映射。

当前主脚本位置：

```text
~/workspace/deepdive_kai0/piper_tools/activate_can.sh
```

`~/huanqian/deepdive_kai0-main/docs/` 下的文档是同步说明文档。除非 huanqian 里的脚本也同步到同一版本，否则实际执行命令建议从 `~/workspace/deepdive_kai0` 项目下运行。

## 为什么要按机器名区分

ROS 启动文件使用稳定的 CAN 名称：

- `can_left_mas`
- `can_right_mas`
- 旧 4-CAN 拓扑还会用到 `can_left_slave`、`can_right_slave`

但 Linux 看到的是 USB CAN 适配器所在的物理 USB bus-info。不同机器人、不同 USB 插法、不同 hub 拓扑下，同一个 CAN 适配器的 bus-info 可能不同。

所以 `activate_can.sh` 会做这几件事：

1. 从 `hostname` 自动判断机器人名字，或者使用 `--machine` / `--robot` 手动指定。
2. 选择对应机器的 USB bus-info 映射。
3. 把匹配到的 SocketCAN 接口重命名成 ROS 期望的符号名。
4. 以 `1000000` bitrate 拉起 CAN。
5. 检查启用的 CAN 口是否能收到数据帧。

## 机器人映射

### visrobot02

当前机器人，使用 two-CAN 拓扑：左右各一条共享 CAN，每侧 master 和 slave 在同一条 CAN 总线上。

| USB bus-info | CAN 名称 | 作用 |
| --- | --- | --- |
| `1-13:1.0` | `can_left_mas` | 左侧共享 CAN |
| `1-12:1.0` | `can_right_mas` | 右侧共享 CAN |
| `1-1:1.0` | `can_left_slave` | 旧候选口，默认不用 |

在 `visrobot02` 上，不传参数时默认使用 `two-can` 模式。

### visrobot01 / sim01

旧机器，使用 legacy 4-CAN 拓扑。

| USB bus-info | CAN 名称 | 作用 |
| --- | --- | --- |
| `3-2.2.2:1.0` | `can_left_mas` | 左 master |
| `3-2.2.1:1.0` | `can_left_slave` | 左 slave |
| `3-2.2.3:1.0` | `can_right_mas` | 右 master |
| `3-2.2.4:1.0` | `can_right_slave` | 右 slave |

在 `visrobot01` / `sim01` 上，不传参数时默认使用 `four-can` 模式。

## 手动激活 CAN

当前机器人 `visrobot02` 推荐命令：

```bash
cd ~/workspace/deepdive_kai0
SUDO_PASSWORD=agx bash piper_tools/activate_can.sh --robot visrobot02 --two-can
```

如果当前主机名就是 `visrobot02`，也可以不写机器名：

```bash
cd ~/workspace/deepdive_kai0
SUDO_PASSWORD=agx bash piper_tools/activate_can.sh
```

旧机器 `visrobot01`：

```bash
cd ~/workspace/deepdive_kai0
SUDO_PASSWORD=agx bash piper_tools/activate_can.sh --robot visrobot01 --four-can
```

也可以用环境变量指定机器：

```bash
cd ~/workspace/deepdive_kai0
KAI0_ROBOT_ID=visrobot02 SUDO_PASSWORD=agx bash piper_tools/activate_can.sh
```

也可以使用新的统一变量名：

```bash
VIS_ROBOT_ID=visrobot02 SUDO_PASSWORD=agx bash piper_tools/activate_can.sh
```

## 自主推理启动

`~/workspace/deepdive_kai0/start_scripts/kai/start_autonomy.sh` 已经在 Step 3 中调用 workspace 自己的 `piper_tools/activate_can.sh`。

因此现在启动本地自主推理时，不需要再先到 huanqian 目录手动激活 CAN。

当前 `visrobot02` 的常用命令：

```bash
cd ~/workspace/deepdive_kai0
KAI0_VENV=/home/agilex/workspace/deepdive_kai0/kai0/.venv_humble310 \
SUDO_PASSWORD=agx \
./start_scripts/kai/start_autonomy_from_ckpt.sh \
  checkpoints/ckpt_v0/awbc_step49999 \
  --no-rerun \
  record_enable:=false
```

如果主机名不可靠，可以显式指定：

```bash
cd ~/workspace/deepdive_kai0
KAI0_ROBOT_ID=visrobot02 \
KAI0_VENV=/home/agilex/workspace/deepdive_kai0/kai0/.venv_humble310 \
SUDO_PASSWORD=agx \
./start_scripts/kai/start_autonomy_from_ckpt.sh \
  checkpoints/ckpt_v0/awbc_step49999 \
  --no-rerun \
  record_enable:=false
```

## 验证方法

### 1. 看激活输出

```bash
cd ~/workspace/deepdive_kai0
SUDO_PASSWORD=agx bash piper_tools/activate_can.sh --robot visrobot02 --two-can
```

正常输出应该包含：

```text
robot: visrobot02 (visrobot02 two-CAN)
mode : two-can (left/right shared CAN)
[OK] can_tmp_0 -> can_left_mas (bus: 1-13:1.0)
[OK] can_tmp_1 -> can_right_mas (bus: 1-12:1.0)
can_left_mas: OK (has data)
can_right_mas: OK (has data)
```

### 2. 看当前 CAN 口对应哪个 USB 位置

```bash
ip -details link show can_left_mas can_right_mas
readlink -f /sys/class/net/can_left_mas/device
readlink -f /sys/class/net/can_right_mas/device
```

`visrobot02` 上应看到：

```text
can_left_mas  -> 1-13:1.0
can_right_mas -> 1-12:1.0
```

### 3. 看 ROS 关节反馈

只读检查，不会发动作：

```bash
source /opt/ros/humble/setup.bash
source ~/workspace/deepdive_kai0/ros2_ws/install/setup.bash
ros2 topic hz /puppet/joint_left
ros2 topic hz /puppet/joint_right
```

正常情况下 Piper 关节反馈大约是 `200Hz`。

## 常见问题

### unknown robot

如果脚本提示 `unknown robot`，说明 hostname 不在脚本已知列表里。可以手动指定：

```bash
bash piper_tools/activate_can.sh --machine visrobot02 --two-can
```

或者：

```bash
export VIS_ROBOT_ID=visrobot02
```

### CAN UP 但没有数据

常见原因：

- 机械臂没上电。
- 急停按钮被按下。
- CAN 线没插好。
- USB CAN 插到了错误位置。
- USB hub 或控制器处在异常状态。
- 旧 ROS/CAN 进程残留，占用了设备或造成 DDS 状态混乱。

先检查：

```bash
ip -details link show can_left_mas can_right_mas
timeout 2 candump can_left_mas -n 1
timeout 2 candump can_right_mas -n 1
```

### 换 USB 口之后异常

这套映射依赖 USB bus-info。换 USB 口后，CAN 适配器可能会出现在新的 bus-info 上，因此旧映射可能不再正确。

换口后建议记录：

```bash
for i in can_left_mas can_right_mas can0 can1 can2 can3; do
  [ -e /sys/class/net/$i ] || continue
  echo "### $i"
  ip -details link show "$i" | grep parentdev || true
  readlink -f "/sys/class/net/$i/device"
done
```

如果 USB 物理布局以后固定改变，就要更新 `piper_tools/activate_can.sh` 中对应机器的映射。

### 左右臂动作看起来反了

不要第一时间交换 action 维度。优先按下面顺序查：

1. `activate_can.sh` 是否把 `can_left_mas` 和 `can_right_mas` 映射到预期 USB bus-info。
2. 手动轻微移动物理左臂时，是否是 `/puppet/joint_left` 变化。
3. 手动轻微移动物理右臂时，是否是 `/puppet/joint_right` 变化。
4. 只有确认硬件反馈左右正确后，再查 policy observation / action 的左右顺序。

## 新增另一台机器人

在 `piper_tools/activate_can.sh` 里新增一个 `case` 分支即可：

```bash
new_robot_name)
    ROBOT_DESC="new robot description"
    DEFAULT_TOPOLOGY="two-can"
    SLAVE_MAPPINGS=(
        "bus-info:can_left_slave"
    )
    MASTER_MAPPINGS=(
        "bus-info:can_left_mas"
        "bus-info:can_right_mas"
    )
    ;;
```

然后验证：

```bash
SUDO_PASSWORD=agx bash piper_tools/activate_can.sh --robot new_robot_name
```
