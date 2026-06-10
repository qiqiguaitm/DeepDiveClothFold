#!/usr/bin/env python3
"""Submit an AIHC aijob spec via raw OpenAPI POST, reusing the `aihc` CLI's BCE signing.

Why not `aihc job create`/`job submit`: our AIHC access is a **granted queue on the shared
`aihc-serverless` pool**, not an owned pool. The CLI resolves pool name->numeric id
(`get_pool_id_by_name`) which 404s for serverless, so its create/list/get are unusable here
(same finding as dreamzero). Instead we POST straight to /api/v1/aijobs?resourcePoolId=<pool>
with the pool *name* used directly as the id — verified working.

Auth (AK/SK/host) and the image-pull cred (username/password) are read from ~/.aihc/config
(populated by `aihc config --ak .. --sk .. --username .. --password ..`); nothing secret is
read from or written to the repo.

Usage:
  python scripts/aihc/submit_raw.py <spec.json> [--pool aihc-serverless] [--dry]
"""
import argparse
import copy
import json
import sys

from aihc_cli_py.aihc_argumentparser import config_file
from aihc_cli_py.configure import get_ak_sk, get_username_password
from aihc_cli_py.utils import send_request


def _walk(o, fn):
    if isinstance(o, dict):
        fn(o)
        for v in o.values():
            _walk(v, fn)
    elif isinstance(o, list):
        for v in o:
            _walk(v, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="path to aijob spec json (e.g. scripts/aihc/aijob_abs_lookahead_5n8g.json)")
    ap.add_argument("--pool", default="aihc-serverless")
    ap.add_argument("--queue", default=None, help="queueID (serverless pool requires it); default = spec's 'queue' field")
    ap.add_argument("--dry", action="store_true", help="print the body (password redacted) and exit; do not submit")
    args = ap.parse_args()

    ak, sk, host = get_ak_sk(config_file)
    if not ak or not sk:
        sys.exit("no AK/SK in ~/.aihc/config — run: aihc config --ak <id> --sk <secret> --region cn-beijing")
    user, pw = get_username_password(config_file)

    body = json.load(open(args.spec))
    # the raw /aijobs endpoint wants the enum value (the repo spec uses the Go-CLI `-f` lowercase form)
    _FW = {"pytorch": "PyTorchJob", "pytorchjob": "PyTorchJob", "mpi": "MPIJob", "mpijob": "MPIJob"}
    if "jobFramework" in body:
        body["jobFramework"] = _FW.get(str(body["jobFramework"]).lower(), body["jobFramework"])
    # inject image-pull cred from config into every imageConfig (kept out of the repo spec)
    def _inject(d):
        ic = d.get("imageConfig")
        if isinstance(ic, dict):
            if user:
                ic["username"] = user
            if pw:
                ic["password"] = pw
    _walk(body, _inject)

    # serverless pool requires queueID as a query param (the body's "queue" field is ignored)
    queue = args.queue or body.get("queue", "")
    url = f"http://{host}/api/v1/aijobs?resourcePoolId={args.pool}&queueID={queue}"

    if args.dry:
        red = copy.deepcopy(body)
        _walk(red, lambda d: d.get("imageConfig", {}).update(password="<redacted>")
              if isinstance(d.get("imageConfig"), dict) and d["imageConfig"].get("password") else None)
        print("POST", url)
        print(json.dumps(red, ensure_ascii=False, indent=2))
        return

    r = send_request(url, "post", ak, sk, json=body)
    res = r.get("result", r) if isinstance(r, dict) else r
    job_id = res.get("jobId") or res.get("jobName") if isinstance(res, dict) else None
    print("submitted:", json.dumps(r, ensure_ascii=False))
    print("jobId:", job_id)


if __name__ == "__main__":
    main()
