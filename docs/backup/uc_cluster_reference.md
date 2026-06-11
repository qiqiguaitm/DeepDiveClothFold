# uc 集群参考 (归档) — 从 active 文档剪出的 uc 段

> uc01/02/03 已**彻底停用**(2026-05-18 重装后退役)。本文件汇集原先散落在 `deployment/training_ops/storage_and_env.md` 与 `ssh_and_credentials.md` 里的 uc 专属段落,供查史。**路径/IP/拓扑均已失效。**

---

## A. 数据集源 (原 storage_and_env.md §2.4)

#### uc01 / uc02 / uc03 (NFS 共享 — uc01 export 给 uc02/03; 2026-05-28 已迁到 kai0/data 对齐 gf0)

```
/data/shared/ubuntu/workspace/deepdive_kai0/kai0/data/Task_A/   # 与 gf0/gf3 同规范; uc01 NFS export → uc02/03 共享
    kai0_base/ kai0_dagger/ kai0_advantage/  # HF 官方 (3055/3457/3055) 真实目录 (原 dataset/Kai0_official/Task_A/*)
    vis_base/                  # build 源; 2026-06-02 v2/v3 分层: v2/<date>-v2 ×20 (含depth, sync DST) + v3/<date>-v3 ×20 (裁投放, 无depth)
    vis_dagger/ vis_autonomy/ vis_inference/  # uc 特有: 原始自采 dagger / autonomy(98G) / inference
    self_built/                # 所有构建集 (vis_v2_merged 自包含真实, xvla_exp1 视频软链, A_new_100 软链 vis_base)
```

> **变更要点 (2026-05-28 迁移)**: uc 原本数据在独立的 `/data/shared/ubuntu/workspace/dataset/` 树, 与 gf0/gf3 的 `kai0/data/` 不一致 → 已全部迁入 `kai0/data/Task_A/` 并对齐命名; 跨数据集软链 (22647 条) 已 retarget, dangling=0; 只在 uc01 操作, uc02/03 经 NFSv4.1 自动可见。

---

## B. SSH 互信拓扑 (原 ssh_and_credentials.md §4.4)

### uc 集群 SSH 互信拓扑 (2026-05-18 重装后)

3 台 uc 间 ubuntu 用户 ed25519 互信 (cloud-init pre-seed)。本地 dev (tim) → 3 server ubuntu authorized_keys;uc01/02/03 彼此 ed25519 互信 (6 方向)。

| Host | 内网 | Pubkey 前缀 |
|---|---|---|
| uc01 | 10-60-135-47 | `AAAAC3NzaC1lZDI1NTE5AAAAIF+mEiKsU8Q2fiXWl9fG/6J+THe9+vMZKjvICm0srfLb` |
| uc02 | 10-60-204-66 | `AAAAC3NzaC1lZDI1NTE5AAAAIPOYAi7KHrboT1M1AVXiulnVlyzAmJAa3HKzXaNDfc0n` |
| uc03 | 10-60-253-225 | `AAAAC3NzaC1lZDI1NTE5AAAAILQdFOvow28O9HalNIPUCElD/im+FHxQCiP9N2yVtWYD` |

> 测试: `ssh uc01 hostname` (本地→uc01) / `ssh uc01 'ssh ubuntu@10.60.204.66 hostname'` (uc01→uc02)。uc ubuntu key 未推到 gf/sim;跨集群 SSH 走 tim 用户旧互信。
