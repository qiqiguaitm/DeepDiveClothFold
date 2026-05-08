"""Download a file from Volcano TOS (transfer-shanghai bucket) with multi-part + progress."""
import os
import argparse
import time

for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import tos

ak = os.environ.get("VOLC_TOS_AK") or os.environ.get("TOS_ACCESS_KEY")
sk = os.environ.get("VOLC_TOS_SK") or os.environ.get("TOS_SECRET_KEY")
if not ak or not sk:
    raise SystemExit("Missing VOLC_TOS_AK / VOLC_TOS_SK env vars (or legacy TOS_ACCESS_KEY / TOS_SECRET_KEY).")
endpoint = "tos-cn-shanghai.volces.com"
region = "cn-shanghai"
bucket_name = "transfer-shanghai"

parser = argparse.ArgumentParser()
parser.add_argument("--object_key", required=True, help="TOS 对象键（含路径，不带桶名）")
parser.add_argument("--file", required=True, help="本地目标文件路径")
parser.add_argument("--task_num", type=int, default=16, help="分片并发数")
parser.add_argument("--part_size_mb", type=int, default=64, help="分片大小 MB")
args = parser.parse_args()

client = tos.TosClientV2(ak, sk, endpoint, region)

head = client.head_object(bucket_name, args.object_key)
total = head.content_length
print(f"远端大小: {total / 1024**3:.2f} GiB  ({total} bytes)", flush=True)

os.makedirs(os.path.dirname(args.file) or ".", exist_ok=True)

if os.path.exists(args.file) and os.path.getsize(args.file) == total:
    print(f"{args.file} 已存在且大小一致，跳过")
    raise SystemExit(0)

last_print = [0.0]
def percentage(consumed_bytes, total_bytes, rw_once_bytes, type):
    now = time.time()
    if total_bytes and (now - last_print[0] > 2 or consumed_bytes == total_bytes):
        rate = 100 * consumed_bytes / total_bytes
        elapsed = now - t0
        mbps = consumed_bytes / 1024**2 / max(elapsed, 1e-6)
        eta = (total_bytes - consumed_bytes) / 1024**2 / max(mbps, 1e-6)
        print(f"progress: {rate:5.2f}%  {consumed_bytes/1024**3:.2f}/{total_bytes/1024**3:.2f} GiB  {mbps:6.1f} MB/s  ETA {eta/60:.1f} min", flush=True)
        last_print[0] = now

print(f"开始下载: {args.object_key} -> {args.file}  (task_num={args.task_num}, part_size={args.part_size_mb} MB)", flush=True)
t0 = time.time()
client.download_file(
    bucket_name, args.object_key, args.file,
    task_num=args.task_num,
    part_size=1024 * 1024 * args.part_size_mb,
    data_transfer_listener=percentage,
)
dt = time.time() - t0
size = os.path.getsize(args.file)
print(f"完成: 耗时 {dt:.1f}s, 平均 {size/1024/1024/dt:.2f} MB/s")
