# 2026-05-16 — uc 集群挖矿木马入侵事件 + 重装备份记录

> **事件概要**: 2026-05-15 ~ 16, uc01/02/03 三节点被 SSH 密码爆破入侵, 植入 Ravencoin 挖矿木马 (Rigel v1.23.1), 训练吞吐降至 1/5~1/10。10:30 矿机全部 kill, 当日完成 uc 重装前关键 ckpt + dataset 备份到 TOS (45.07 GB / 1013 objects)。
> **严重级别**: P0。
> **报告时间**: 2026-05-16 (合并自原 `docs/security/` 两个文档)。

---

# Part I — 安全事件: Ravencoin 挖矿木马入侵

**事件日期**: 2026-05-15 ~ 2026-05-16
**发现时间**: 2026-05-16 ~10:00 CST
**影响范围**: uc 集群全部 3 个 GPU 节点 (uc01, uc02, uc03)
**处置时间**: 2026-05-16 10:30 CST (矿机进程已 kill, 证据保留)

## 1. Executive Summary

攻击者通过 **SSH 密码爆破** 入侵 tim 账户, 在 uc01/02/03 三个生产 GPU 节点上部署 **Ravencoin (RVN) 加密挖矿木马** (Rigel v1.23.1, 路径伪装为 `/var/tmp/systemd-private-*-ModemManager.service-*/python/`)。矿机持续运行 8+ 小时, 占用每节点 1 张 GPU 6.1 GB 显存 + 100% 计算, 导致并行 pi05 训练吞吐**降至原速 1/5 ~ 1/10**, 多次 inline_eval 卡停。矿机已于 10:30 全部 kill, 但**密码爆破入侵向量仍有效**, 必须修密码 + 禁用密码登录 + 审计其他节点。

## 2. 攻击时间线

### 入侵阶段
| 时间 (CST) | 节点 | 事件 |
|---|---|---|
| 2026-05-15 22:19:07 | uc01 | 攻击者第一次 SSH 密码登录 (`tim@172.104.235.108`, port 49435) |
| 2026-05-15 22:19~00:43 | uc01 | session 1, 持续 2h24m (推测扫描+准备) |
| 2026-05-16 00:22 | uc02 | 矿机目录创建 |
| 2026-05-16 00:28 | uc02 | 矿机首次启动 (PID 142542 watchdog + 142772 worker) |
| 2026-05-16 00:43:15 | uc01 | session 2 (17min) |
| 2026-05-16 01:03:22 | uc01 | session 3 (持续到 02:02:50) |
| 2026-05-16 01:12 | uc03 | 矿机启动 (PID 3774844 watchdog + 3775074 worker, stratum localhost:1700) |
| 2026-05-16 01:59:58 | uc01 | 矿机首次启动 (尝试直连 `5.35.103.246:12222`, CTRL+C 异常退出) |
| 2026-05-16 02:00:15 | uc01 | 矿机重启 (改用 `localhost:1505` 本地代理) |
| **2026-05-16 02:02:50** | uc01 | **攻击者 SSH session 3 结束** (恰好对应矿机稳定运行后) |

### 检测/响应阶段
| 时间 (CST) | 事件 |
|---|---|
| 2026-05-16 10:00 | tim 团队发现 uc02/uc03 训练 stuck 8h, 开始排查 |
| 10:18 | `nvidia-smi --query-compute-apps` 发现 GPU 上有非训练 python 进程 |
| 10:22 | 确认 3 节点全部感染同一矿机 (相同钱包 `RCNQT8z...`) |
| 10:27-10:30 | 三节点矿机全部 kill (先 watchdog 后 worker) |
| 10:32 | sshd auth.log 确认入侵 vector = SSH 密码爆破 |

## 3. 攻击向量

### 3.1 入侵方式: SSH 密码爆破

**uc01 sshd auth.log 证据**:
```
May 15 22:19:07 ... sshd[272806]: Accepted password for tim from 172.104.235.108 port 49435 ssh2
May 16 00:43:15 ... sshd[346058]: Accepted password for tim from 172.104.235.108 port 49437 ssh2
May 16 01:03:22 ... sshd[699263]: Accepted password for tim from 172.104.235.108 port 49438 ssh2
```

> ⚠️ **`Accepted password`** = 密码登录成功 (vs `Accepted publickey` 正常 key 登录)。tim 团队自己访问全部 publickey + RSA SHA256:wqWeeyNo... (`14.103.218.231` 客户端 IP), 从未使用密码。

### 3.2 攻击源 IP: `172.104.235.108`

- 地理: 美国, **Linode** (linode.com) 云数据中心 — 攻击者租用的 VPS 跳板
- 已知的密码爆破/挖矿木马传播源 IP 类别

### 3.3 SSH key 未被泄漏

`/home/tim/.ssh/authorized_keys` 共 7 个 key, 经 comment 字段确认全为 tim 团队所有。**入侵 100% 通过密码, 而非 key**。

## 4. 部署的木马详情

### 4.1 矿机进程结构

```
parent (watchdog):  ./python -a kawpow --coin rvn -o stratum+tcp://localhost:<PORT> \
                          -u RCNQT8zuwq466dbAnJaKkWuwzdXTXGbKrg -w rig --log-file logs/miner.log
child (worker):     /var/tmp/systemd-private-<UUID>-ModemManager.service-<RAND>/python/python \
                          -a kawpow ... --watchdog-pid <PARENT_PID>
```

**Watchdog 机制**: 父监控子, 子被 kill 自动复活 → **必须先 kill 父进程**。

### 4.2 目录结构 (各节点一致)

```
/var/tmp/systemd-private-<UUID>-ModemManager.service-<RAND>/python/
├── python      131,798,144 B (Rigel v1.23.1 二进制, 重命名)
├── rvn.sh             505 B (启动脚本)
├── end                 39 B (用途未明)
└── logs/miner.log
```

文件 mtime: `python` 2025-12-24 04:56 / `rvn.sh` 2026-04-17 03:06 / `end` 2026-03-11 21:48 (各节点一致 → 同一份木马包)。

### 4.3 钱包 + 矿池

| 项 | 值 |
|---|---|
| **钱包地址** | `RCNQT8zuwq466dbAnJaKkWuwzdXTXGbKrg` (Ravencoin, 三节点共用) |
| Worker | `rig` (三节点共用) |
| **真实矿池** | `5.35.103.246:12222` (推测 2miners EU server) |
| 本地代理 | `localhost:1504` (uc01) / `:1400` (uc02) / `:1700` (uc03) |
| 算法 | KAWPOW (RVN 专用 PoW) |

> 本地 stratum 代理是攻击者搭建的中转层, 用于隐藏真实矿池流量 (绕过出站防火墙规则)。

### 4.4 矿机二进制哈希

```
uc01: 6cca93f05bff87ad6d491531abefdd07441446f67629a0f03d8488fe651dd7e1
uc02/03: (取证文件已保留, hash 待确认)
```

## 5. IoCs

- **IP**: `172.104.235.108` (攻击源), `5.35.103.246` (矿池, 真实出站)
- **本地端口**: 1400, 1504, 1700
- **目录**: `/var/tmp/systemd-private-*-ModemManager.service-*/python/`
- **文件**: `python` (131 MB), `rvn.sh`, `end`
- **二进制 hash**: `6cca93f05bff87ad6d491531abefdd07441446f67629a0f03d8488fe651dd7e1`
- **进程命令行**: `kawpow`, `--coin rvn`, `RCNQT8zuwq466dbAnJaKkWuwzdXTXGbKrg`, `stratum+tcp`, `--watchdog-pid`
- **Wallet**: `RCNQT8zuwq466dbAnJaKkWuwzdXTXGbKrg` (raven.tokens.fyi / ravencoin.cc 可查)

## 6. 已保留的证据

各节点 `/home/tim/miner_evidence_<YYYYMMDD_HHMM>/`:
```
uc01_rvn.sh             — 启动脚本 (505 B)
uc01_end                — 用途未明 (39 B)
uc01_miner_log_head.txt — 矿机首 200 行日志
uc01_python_sha256.txt  — Rigel 二进制 hash
uc01_logs_ls.txt        — logs/ 目录列表
```
uc02/03 类似 (uc03 chmod 限制部分文件未取到, hash 保留)。

## 7. 即时处置 (已完成)

| 时间 | 节点 | 处置 |
|---|---|---|
| 10:27 | uc01 | `kill -9 3456907 3457047` (watchdog + worker) |
| 10:29 | uc02 | `kill -9 142542 142772` |
| 10:30 | uc03 | `kill -9 3774844 3775074` |
| 10:32 | 三节点 | 验证 GPU compute-apps 仅剩训练 ✓ |

> 注: 仅 kill 当前 instance, 未 disable autostart 机制 (尚未确认是否有 cron 重启)。

## 8. Action Items

### 🔴 P0 (立即, 防止再次入侵)
- [ ] **修改 tim 密码** (uc01-03 / js01-04 / gf* / ipc01 所有节点): 强密码 ≥16 char 含大小写数字符号
- [ ] **禁用 SSH 密码登录** (所有节点): `sshd_config: PasswordAuthentication no` + `systemctl reload sshd`
- [ ] **重启 sshd**

### 🟠 P1 (今日, 防扩散 + 完整审计)
- [ ] 审计 uc02 / uc03 auth.log 查找 `172.104.235.108`
- [ ] 审计 js 集群 (js01-04) 是否同样感染 (`/var/tmp/systemd-private-*` + auth.log)
- [ ] 审计 gf 集群 (gf4 等) 是否被入侵
- [ ] 审计 ipc01 是否被入侵
- [ ] 检查 `~/.bash_history` 看攻击者在 uc01 执行了哪些命令
- [ ] 检查 cron / systemd timer / autostart 持久化机制
- [ ] 检查 /etc/passwd, /etc/sudoers, /etc/cron.d/ 是否有未识别修改

### 🟡 P2 (本周, 加固)
- [ ] 安装 fail2ban (`apt install fail2ban`) 自动 ban 多次失败 SSH
- [ ] 云防火墙白名单 (Tencent CVM 控制台) 限制 22 端口源 IP
- [ ] sshd `MaxAuthTries=3` + `LoginGraceTime=10s`
- [ ] rotate 所有 SSH key (密码已泄漏账户对应 key 也宜 rotate)
- [ ] rotate wandb token, 检查 tos 等云对象存储凭证

### 🟢 P3 (本月, 长期)
- [ ] 部署集中日志 (auditd / syslog 中央服务器)
- [ ] 评估 wallet 链上活动, 看攻击者其他受害者规模
- [ ] 提交 abuse 报告给 Linode (针对 172.104.235.108)
- [ ] 文档化为 runbook: 训练 SOP 加入"启动训练时检查 GPU compute-apps 是否有非训练进程"

## 9. 训练影响评估

### 受影响训练 (8h+ 严重降速)
| 训练 | 节点 | 正常 rate | 受感染 rate | 退化倍数 | 当前 step |
|---|---|---|---|---|---|
| pure_1800_mixed1 | uc02 | 1.9 s/it | 14-15 s/it | 7.5× | 40k/50k |
| smooth_800 (nw=64) | uc03 | 1.9 s/it | 17-18 s/it | 9.4× | 40k/50k |
| pi05init (pi05_base) | uc01 | 期望 1.9 s/it | 25 s/it | 13× | 5.7k/50k (已 kill) |

### ckpt 完整性
- 已 finalize 的 ckpt (step 40000 之前所有): 未受影响, 可用
- step 40000 ckpt: 已 finalize, 训练 stuck 在 inline_eval, ckpt 本身完整

## 10. 经验教训

1. **不要启用 SSH 密码登录** — 这是 90% 入侵 GPU 服务器的来源。**Only SSH keys**。
2. **监控 GPU 上非授权进程** — 启动训练时 `nvidia-smi --query-compute-apps`, 非自己启动的 python 立即查证。
3. **`/var/tmp/systemd-private-*-ModemManager.service-*`** 路径是常见挖矿木马伪装。日常 audit 扫这种路径。
4. **`watchdog + worker` 进程对** 是矿机标准结构, kill 时必须先停 watchdog。
5. **本地 stratum 代理 (localhost:port)** 是隐藏出站连接的常用手法 — 出站防火墙看实际 destination IP, 而非协议端口。

## 11. 参考

- Rigel v1.23.1: https://github.com/rigelminer/rigel (合法开源 GPU 矿机, 被恶用)
- 2miners pool: https://rvn.2miners.com
- Wallet explorer: https://raven.tokens.fyi/wallet/RCNQT8zuwq466dbAnJaKkWuwzdXTXGbKrg

---

# Part II — uc 重装前备份清单 (Backup Manifest)

**备份日期**: 2026-05-16
**TOS 路径根**: `tos://transfer-shanghai/backup_uc_reinstall_20260516/`
**总大小**: ~45.2 GB / 1013 objects
**备份完成**: 2026-05-16 19:18 CST ✅
**实测速度**: 135 MB/s (cron `pull_tos_to_shared.sh` 已禁用避免抢带宽)
**TOS 验证**:
- `datasets/A_new_pure_200/`: **815 objects, 3.14 GB** ✓
- `datasets/A_new_pure_200_val/`: **94 objects, 262.74 MB** ✓ (补传 2026-05-16 19:23 CST)
- `ckpts/pi05init_step_4000/`: **104 objects, 41.67 GB** ✓

## 12. 备份内容清单

| # | 对象 | 大小 | TOS 路径 | 原 uc 路径 | 用途 |
|---|---|---|---|---|---|
| 1 | **uc01 pi05init ckpt step 4000** | 42 GB | `tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/pi05init_step_4000/` | `uc01:/home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/4000/` | **resume 训练 (paused at step 4330)** |
| 2 | **A_new_pure_200 dataset** | 3.2 GB | `.../datasets/A_new_pure_200/` | `uc01:/home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200/` | resume 训练必需数据 |
| 3 | **A_new_pure_200_val** (inline_eval) | 263 MB | `.../datasets/A_new_pure_200_val/` | `uc01:/home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200_val/` | inline_eval 必需 |

## 13. 实验上下文 (paused state)

| 参数 | 值 |
|---|---|
| Config name | `pi05_flatten_fold_a_new_pure_200_js` |
| Exp name | `task_a_pure200_new_norm_base_pi0.5` |
| Init | pi05_base (raw pretrained) |
| Init 路径 (原) | `uc01:/home/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params` |
| Data | A_new_pure_200 (200 ep `-new` 精选 + hflip mirror) |
| Steps target | 50,000 |
| **Steps reached** | **4330** (latest ckpt 在 step 4000) |
| Batch | 120, FSDP=8 |
| num_workers | 64 (uc 单机 + 本地 SSD 推荐) |
| LR | 1.5e-5 → 1.5e-6 cosine, warmup 1k |
| EMA | 0.9999 |
| WandB | offline |
| Status | **paused 2026-05-16 18:36 CST** (用户主动停止, ckpt 4000 完整) |

**已有 step 4000 eval**: MAE@1=0.0507 / @10=0.0663 / @25=0.0927 / @50=0.1274 (pi05_base init 起点, 与 pure2_1800_6000 step 4k=0.0534 相近, 符合预期)

## 14. uc 重装后 resume 流程

### Step 1: 从 TOS 拉回数据 + ckpt
```bash
# 1. dataset
mkdir -p /home/tim/local_ckpts/data/Task_A/self_built/
tosutil cp -r tos://transfer-shanghai/backup_uc_reinstall_20260516/datasets/A_new_pure_200/ \
  /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200/ -j 32 -p 8

# 2. ckpt step 4000
mkdir -p /home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/
tosutil cp -r tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/pi05init_step_4000/ \
  /home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/4000/ -j 32 -p 8

# 3. 验证大小 (应 3.2G + 42G)
```

### Step 2: 验证 ckpt 完整性
应看到 `_CHECKPOINT_METADATA / assets/ / params/ / train_state/`。**缺 `train_state/` → resume 无法工作** (需优化器状态)。

### Step 3: 准备 pi05_base init (未备份)
```bash
# pi05_base 在备份中未包含 (12G), 从 HF 重下:
mkdir -p /home/tim/workspace/openpi_cache/openpi-assets/checkpoints/
huggingface-cli download openpi-assets/pi05_base \
  --local-dir /home/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base
```

### Step 4: 预防 cron 拖慢 (重要!)
```bash
# resume 前禁掉 pull_tos_to_shared.sh cron, 否则训练 rate 退化到 5.5 s/it:
crontab -l > /tmp/tim_cron_backup_$(date +%Y%m%d).txt
crontab -l | sed 's|^\(\*/5 \* \* \* \* /home/tim/scripts/pull_tos_to_shared.sh\)|#\1|' | crontab -
```

### Step 5: 启动 resume 训练
```bash
cd /data/shared/tim/workspace/deepdive_kai0/kai0
nohup env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  OPENPI_DATA_HOME=/home/tim/workspace/openpi_cache \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  JAX_COMPILATION_CACHE_DIR=/home/tim/workspace/xla_cache_uc01 \
  XLA_FLAGS=--xla_gpu_autotune_level=0 \
  .venv/bin/python -u scripts/train.py pi05_flatten_fold_a_new_pure_200_js \
    --exp-name task_a_pure200_new_norm_base_pi0.5 \
    --batch-size 120 --fsdp-devices 8 --num-workers 64 \
    --data.repo-id /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200 \
    --inline-eval-val-root /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200_val \
    --weight-loader.params-path /home/tim/workspace/openpi_cache/openpi-assets/checkpoints/pi05_base/params \
    --checkpoint-base-dir /home/tim/local_ckpts/checkpoints \
    --no-wandb-enabled --resume \
  > /data/shared/tim/logs/train_task_a_pure200_new_norm_base_pi0.5_uc01.log 2>&1 < /dev/null & disown -a
```

**关键 flag**: `--resume` (不要用 `--overwrite`, 会从 step 0 重启)。

### Step 6: 验证恢复
```bash
tail -f /data/shared/tim/logs/train_task_a_pure200_new_norm_base_pi0.5_uc01.log
# 应显示 "Restoring checkpoint from .../4000"
# 预期 ~3 min 编译 → rate ≈ 1.9 s/it (本地盘 + nw=64)
# 如 rate ≥ 5 s/it: ps -ef | grep tosutil | grep -v grep   (应为空)
```

## 15. 备份缺失项 + 注意事项

### 未备份 (重新获取方式)
| 项 | 大小 | 重新获取 |
|---|---|---|
| pi05_base init | 12 GB | HuggingFace `openpi-assets/pi05_base` |
| mixed_1 init | 22 GB | 已废弃 (本次用 pi05_base) |
| deepdive_kai0 代码 | (变化) | git clone GitHub |
| .venv Python 环境 | (变化) | `pip install -e .` 重建 |
| **uc02 / uc03 best ckpt** (MAE@1=0.0088/0.0089) | 各 42 GB | 已在 uc02/03 ckpt 49999, 重装会丢. 如需 deploy 应现在补备份 |

如需保留 uc02/03 best SOTA:
```bash
ssh uc02 "tosutil cp -r /cluster_ckpt/checkpoints/pi05_flatten_fold_a_new_pure2_1800/task_a_new_pure_1800_new_norm_base_mixed1/49999 \
  tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/pure_1800_mixed1_step_49999/ -j 32"
ssh uc03 "tosutil cp -r /data/shared/tim/workspace/deepdive_kai0/kai0/checkpoints/pi05_flatten_fold_a_new_smooth_800_new_norm/task_a_new_smooth_800_new_norm/49999 \
  tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/smooth_800_step_49999/ -j 32"
```
(各 42G, 共 84G 额外)

---

**Backup 执行命令** (历史记录):
```bash
ssh uc01 "tosutil cp -r /home/tim/local_ckpts/data/Task_A/self_built/A_new_pure_200 \
  tos://transfer-shanghai/backup_uc_reinstall_20260516/datasets/A_new_pure_200/ -j 32 -p 8"
ssh uc01 "tosutil cp -r /home/tim/local_ckpts/checkpoints/pi05_flatten_fold_a_new_pure_200_js/task_a_pure200_new_norm_base_pi0.5/4000 \
  tos://transfer-shanghai/backup_uc_reinstall_20260516/ckpts/pi05init_step_4000/ -j 32 -p 8"
```
