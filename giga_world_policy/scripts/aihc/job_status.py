#!/usr/bin/env python3
"""AIHC job status/list via raw OpenAPI GET (the CLI's `job get/list` 404 on the serverless
granted queue's pool-name->id lookup). Reuses the `aihc` CLI's BCE signing + ~/.aihc/config.

Usage:
  python scripts/aihc/job_status.py <jobId>     # one job's status
  python scripts/aihc/job_status.py --list      # all jobs on the queue
"""
import argparse
import json

from aihc_cli_py.aihc_argumentparser import config_file
from aihc_cli_py.configure import get_ak_sk
from aihc_cli_py.utils import send_request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id", nargs="?")
    ap.add_argument("--pool", default="aihc-serverless")
    ap.add_argument("--queue", default="aihcq-z4v1apdppzwy")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    ak, sk, host = get_ak_sk(config_file)

    if args.list or not args.job_id:
        url = f"http://{host}/api/v1/aijobs?resourcePoolId={args.pool}&queueID={args.queue}&pageSize=30&pageNo=1"
        r = send_request(url, "get", ak, sk)
        jobs = r.get("result", {}).get("jobs", []) if isinstance(r, dict) else []
        for j in jobs:
            print(f"{j.get('status','?'):12s} {j.get('jobName','?'):40s} {j.get('jobId','?')}  {j.get('createdAt','')}")
        return

    url = f"http://{host}/api/v1/aijobs/{args.job_id}?resourcePoolId={args.pool}&queueID={args.queue}"
    r = send_request(url, "get", ak, sk)
    res = r.get("result", r) if isinstance(r, dict) else r
    if isinstance(res, dict):
        for k in ("jobName", "jobId", "status", "createdAt", "queue", "priority"):
            if k in res:
                print(f"{k:12s}: {res[k]}")
        for k in ("phase", "message", "reason", "subState"):
            if res.get(k):
                print(f"{k:12s}: {res[k]}")
    else:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
