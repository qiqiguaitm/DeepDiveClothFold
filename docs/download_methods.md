# 下载方法(数据 / 模型)— di-* 本机实测(2026-07-09)

> **给所有 agent**:在 di-* 本机(`di-20260312174527-*`)下载 HF 模型 / pip 包 / 数据集时,**默认走本地网络 + 域内镜像,不走代理**。代理(sim01 `127.0.0.1:29290`、clash `7890`、gsy `17897`)对**国际站反而慢**(HF 走任何代理节点都只有 20–600 B/s,连 mihomo 多节点/Global 模式都慢)。本地网络直连域内镜像快 1000×。

## 铁律:下载前完全清代理

curl 即使 `http_proxy` unset,只要 `ALL_PROXY`/`all_proxy` 还在就仍走代理。必须**全清**:
```bash
clean(){ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy "$@"; }
# 或脚本: proxy-off (定义在 ~/.proxy_env)
```

## 实测速度(本地网络无代理)

| 目标 | 速度 | 说明 |
|---|---|---|
| aliyun pip 镜像 | **6 MB/s** | pip/uv 用它 |
| hf-mirror.com CDN(经 huggingface_hub) | **~10 MB/s** | HF 模型/数据用它 |
| huggingface.co 直连 | **0(被墙)** | 不可用 |
| hf-mirror 裸 curl resolve URL | 891 B/s | ⚠️别用裸 curl,会重定向到慢后端 |
| 任何代理下 HF | 20–600 B/s | ⚠️代理对国际站慢 |

## 1) HF 模型 / 数据集 → hf-mirror + huggingface_hub

```bash
clean HF_ENDPOINT=https://hf-mirror.com python -c "
import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('<owner>/<repo>', local_dir='<dst>', max_workers=8,
                  allow_patterns=['params/*','assets/*'])   # 按需 allow_patterns
"
```
- **必须用 `huggingface_hub`**(它正确走 hf-mirror CDN);裸 `curl` resolve URL 会重定向到慢后端。
- `HF_ENDPOINT=https://hf-mirror.com` + 清代理,`max_workers=8` 并行。6GB 约 10min。

## 2) pip / uv → aliyun 镜像

```bash
clean UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/" UV_HTTP_TIMEOUT=300 uv sync
clean pip install -i https://mirrors.aliyun.com/pypi/simple/ <pkg>
```
- ⚠️ 个别包在镜像/PyPI **无当前版本 wheel、只有 sdist**(如 `av==14.4.0`)→ 源码编译又缺 dev 库(root 权限)。修法:`pyproject.toml` 加 `[tool.uv] override-dependencies = ["av==14.2.0"]` 降到有 wheel 的版本(av 只做视频解码,推理多半不用)。查有 wheel 的版本:`curl https://mirrors.aliyun.com/pypi/simple/<pkg>/ | grep cp311.*x86_64.whl`。

## 3) 备用:gf3 aria2c(gf3 可用时)

gf3(`root@124.174.16.237:7888`,密码 tim)网络好 + 有 aria2c:主机解析 HF 签名 URL → gf3 `aria2c -x16` 直连下 → rsync 回传。**⚠️ gf3 常磁盘满/中断,用前先 `df -h` 查空间**;满了就用方法 1。

## 4) 域内模型平台

ModelScope(阿里,域内快如 aliyun)有很多模型镜像,但**小型研究上传(如社区 RoboTwin openpi ckpt)通常只在 HF** → 用方法 1。

---
其余踩坑见 memory `reference_local_download_bypass_proxy.md`(Claude Code 会自动加载)。RoboTwin/sim 环境配置见 `lmwm/docs/MASTER_PLAN_lmwm_vla_2026-07.md` 附录 A。
