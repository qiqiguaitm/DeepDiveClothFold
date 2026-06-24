# SSH / 用户 / 凭据 / TOS

> SSH 速查 (各机连接命令) / 用户体系 / TOS 凭据与 bucket / uc 集群 SSH 互信拓扑。
>
> **同 series**: `overview.md` / `storage_and_env.md` / `data_sync_tos.md` / `submission/`

---

## 4. 连接方式 / 用户信息

### 4.1 SSH 速查

```bash
# gf0 (从 sim01 / 任意公网机)
ssh -p 55555 tim@14.103.44.161   # gf0 (反向隧道经 14.103.44.161 跳板)

# gf3 (火山华北 H20 单卡机, root 直连)
ssh -p 7888 root@124.174.16.237  # gf3, 密码 tim (建议改 key-based)

# uc01 / uc02 / uc03 (2026-05-18 重装后, 直连, ubuntu 账户 key-based)
ssh ubuntu@117.50.196.104   # uc01
ssh ubuntu@106.75.68.254    # uc02
ssh ubuntu@117.50.217.231   # uc03
# (旧: sshpass -p tim ssh tim@... — 已废弃, tim 用户在 uc 上不存在)

# 也可在 ~/.bashrc 设别名:
alias gf3='ssh -p 7888 root@124.174.16.237'
alias uc01='ssh ubuntu@117.50.196.104'   # 2026-05-18 后, key-based, 无需密码
alias uc02='ssh ubuntu@106.75.68.254'
alias uc03='ssh ubuntu@117.50.217.231'
```

### 4.2 用户

- **gf0/sim01**: 用户名 `tim`, 密码 `tim` (有密码 sudo)
- **gf3** (火山华北 H20): 用户名 **`root`**, 密码 `tim`。`/root/code/{README*,demo_project}` 是火山初始 demo, 我们的项目在 `/vePFS-North-E/vis_robot/` 下
- **uc01/02/03** (2026-05-18 重装后): 用户名 **`ubuntu`** (不再创建 tim), key-based 登录, 强密码已设
  - cloud-init pre-seed 了本地 dev pubkey + 团队 key (yihaochen / qiqiguaitm / tim@ipc01 等) 到 `/home/ubuntu/.ssh/authorized_keys`
  - 3 台 uc 间 ubuntu 用户 ed25519 互信已配 (详见 §4.4)
  - **⚠️ 重要安全**: 重装后应立刻**禁 SSH 密码登录** (`PasswordAuthentication no` in `/etc/ssh/sshd_config`) 避免被爆破 (上次事件 2026-05-15 即由此引发, 见 `docs/backup/2026-05-16_uc_security_incident_and_backup.md`)
- gf0: 反向隧道无密码 key-based

### 4.3 TOS 凭据 / Bucket

- Bucket: `transfer-shanghai` @ `tos-cn-shanghai.volces.com` (region `cn-shanghai`)
- 读凭据: hardcoded 在 `train_scripts/kai/data/from_tos_file.py` (公开)
- 写凭据: `VOLC_TOS_AK` / `VOLC_TOS_SK` env vars 或 `tosutil` 配置

> **完整 TOS 数据同步架构 (sim01 是源 → TOS 枢纽 → 各训练服务器) 见 §6**。本节仅记录凭据/bucket 信息。

### 4.4 uc 集群 SSH 互信拓扑  ⚠️ 已停用

> uc01/02/03 已彻底停用 (2026-05-18 退役)。SSH 互信拓扑/pubkey 归档见 [`../../backup/uc_cluster_reference.md`](../../backup/uc_cluster_reference.md)。
