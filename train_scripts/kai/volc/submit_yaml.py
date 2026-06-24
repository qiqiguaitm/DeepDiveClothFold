#!/usr/bin/env python3
"""Submit a volc ml_task YAML via Volcengine Python SDK.

Replaces `volc ml_task submit -c <yaml>` when the volc CLI binary is unavailable.

Usage:
  VOLC_AK=... VOLC_SK=... python submit_yaml.py <yaml_path>

Notes:
  - YAML fields map to API per Volcengine ML Platform OpenAPI (CreateCustomTask).
  - This sends a raw POST via the SDK's universal_api (no nested model wrangling).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import yaml
import volcenginesdkcore
from volcenginesdkmlplatform20240701.api.ml_platform20240701_api import MLPLATFORM20240701Api

# SDK 5.0.27 has a broken response deserializer (KeyError: '.models').
# Monkey-patch the interceptor so create_job() can parse the JSON response itself.
import volcenginesdkcore.interceptor.interceptors.deserialized_response_interceptor as _drm

def _safe_intercept(self, ctx):
    if ctx.request.preload_content:
        try:
            ctx.response.result = json.loads(ctx.response.http_response.data)
        except Exception:
            ctx.response.result = {}
    return ctx

_drm.DeserializedResponseInterceptor.intercept = _safe_intercept


# volc Resource Queue name → (region, queue_id, zone_id)
# Looked up via ListResourceQueues 2026-05-18 (Shanghai) and 2026-05-20 (Beijing).
RESOURCE_QUEUES = {
    # cn-shanghai
    "robot-task":           ("cn-shanghai", "q-20251204185107-fvnpx", "cn-shanghai-a"),  # A100-80G × 28
    "robot-task-4090":      ("cn-shanghai", "q-20260115184225-24r6l", "cn-shanghai-a"),
    "Robot-East-H20":       ("cn-shanghai", "q-20260516104437-2ml4v", "cn-shanghai-e"),  # H20 × N
    "Robot-GPU开发机队列":   ("cn-shanghai", "q-20251205141747-xlxlh", "cn-shanghai-a"),
    "multimodal-task":      ("cn-shanghai", "q-20251215144954-nzlv4", "cn-shanghai-a"),
    "multimodal-task-4090": ("cn-shanghai", "q-20260115184052-k4llg", "cn-shanghai-a"),
    # cn-beijing
    "Robot-North-H20":      ("cn-beijing",  "q-20260516104642-khch9", "cn-beijing-e"),   # H20-SXM5-96G × 56 (7 × ml.hpcpni3ln.45xlarge)
}

# legacy alias for back-compat
RESOURCE_QUEUE_NAME_TO_ID = {name: meta[1] for name, meta in RESOURCE_QUEUES.items()}


def parse_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def build_envs(env_list):
    return [{"Name": e["Name"], "Value": str(e["Value"]), "IsPrivate": bool(e.get("IsPrivate", False))} for e in (env_list or [])]


def build_storages(storage_list):
    out = []
    for s in storage_list or []:
        item = {"Type": s["Type"], "MountPath": s["MountPath"]}
        if s["Type"] == "Vepfs":
            item["Config"] = {"Vepfs": {"Id": s["VepfsId"], "SubPath": s.get("SubPath", "")}}
        elif s["Type"] == "Tos":
            item["Config"] = {"Tos": {"Bucket": s["Bucket"]}}
        out.append(item)
    return out


def build_role_specs(role_list, zone_id):
    out = []
    for r in role_list or []:
        # Two resource modes: fixed flavor (InstanceTypeId) OR custom (FlexibleResourceClaim:
        # Cpu/GpuCount/GpuType/MemoryGiB) — the latter lets you dial CPU below the flavor's bundle
        # (e.g. 108 vCPU instead of hpcpni2.28xlarge's 112) to fit a fragmented node's free quota.
        if r.get("FlexibleResourceClaim"):
            fc = r["FlexibleResourceClaim"]
            claim = {}
            if "Cpu" in fc: claim["Cpu"] = float(fc["Cpu"])
            if "GpuCount" in fc: claim["GpuCount"] = float(fc["GpuCount"])
            if "GpuType" in fc: claim["GpuType"] = fc["GpuType"]
            if "MemoryGiB" in fc: claim["MemoryGiB"] = float(fc["MemoryGiB"])
            if "Family" in fc: claim["Family"] = fc["Family"]
            if "RdmaEniCount" in fc: claim["RdmaEniCount"] = int(fc["RdmaEniCount"])
            resource = {"Type": "Flexible", "FlexibleResourceClaim": claim, "ZoneId": zone_id}
        else:
            resource = {"Type": "Preset", "InstanceTypeId": r["Flavor"], "ZoneId": zone_id}
        out.append({
            "Name": r["RoleName"],
            "Replicas": int(r["RoleReplicas"]),
            "Resource": resource,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("yaml_path")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--region", default=None, help="Override region (default: derived from queue name)")
    args = p.parse_args()

    ak = os.environ.get("VOLC_AK") or os.environ.get("VOLC_ACCESSKEY")
    sk = os.environ.get("VOLC_SK") or os.environ.get("VOLC_SECRETKEY")
    if not ak or not sk:
        sys.exit("ERROR: set VOLC_AK and VOLC_SK env vars")

    cfg = parse_yaml(args.yaml_path)

    qname = cfg["ResourceQueueName"]
    meta = RESOURCE_QUEUES.get(qname)
    if meta is None:
        # Allow direct ResourceQueueId override (skip lookup); still needs explicit region/zone.
        queue_id = cfg.get("ResourceQueueId") or qname
        region   = args.region or os.environ.get("VOLC_REGION", "cn-shanghai")
        zone_id  = cfg.get("ZoneId") or f"{region}-a"
    else:
        region, queue_id, zone_id = meta
        if args.region:
            region = args.region

    # ---- API client ----
    print(f"[submit] queue={qname} region={region} zone={zone_id} queue_id={queue_id}")
    configuration = volcenginesdkcore.Configuration()
    configuration.ak = ak
    configuration.sk = sk
    configuration.region = region
    configuration.client_side_validation = False
    volcenginesdkcore.Configuration.set_default(configuration)

    api = MLPLATFORM20240701Api(volcenginesdkcore.ApiClient(configuration))

    # ---- build request body (dict, sent as-is via SDK serializer) ----
    body = {
        "Name": cfg["TaskName"],
        "Description": cfg.get("Description", ""),
        "ResourceConfig": {
            "ResourceQueueId": queue_id,
            "MaxRuntimeSeconds": int(cfg.get("ActiveDeadlineSeconds", 86400)),
            "Roles": build_role_specs(cfg.get("TaskRoleSpecs"), zone_id),
            # 闲时任务 (preemptible): borrow other queues' idle resources, can be preempted anytime
            # (training resumes from latest ckpt). Set Preemptible: true (+ optional Priority) in yaml.
            **({"Preemptible": True} if cfg.get("Preemptible") else {}),
            **({"Priority": int(cfg["Priority"])} if cfg.get("Priority") is not None else {}),
        },
        "RuntimeConfig": {
            "Framework": cfg.get("Framework", "Custom"),
            "Image": {"Url": cfg["ImageUrl"], "Type": "Prebuild"},
            "Command": cfg["Entrypoint"],
            "Envs": build_envs(cfg.get("Envs")),
        },
        "StorageConfig": {
            "Storages": build_storages(cfg.get("Storages")),
        },
    }

    # caching
    if cfg.get("CacheType"):
        body["StorageConfig"]["CacheType"] = cfg["CacheType"]

    # tags
    if cfg.get("Tags"):
        body["Tags"] = cfg["Tags"]

    print("=== REQUEST BODY ===")
    print(json.dumps(body, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("\n[dry-run] skipping API call")
        return

    print("\n=== submitting ... ===")
    try:
        # Monkey-patched DeserializedResponseInterceptor (top of file) returns dict directly.
        d = api.create_job(body)
        if not isinstance(d, dict):
            print("=== UNEXPECTED RESPONSE TYPE ===", type(d), d)
            sys.exit(1)
        print("=== RESPONSE ===")
        print(json.dumps(d, indent=2, ensure_ascii=False, default=str)[:2000])
        # The volc response shape: {"ResponseMetadata": {...}, "Result": {"Id": "t-..."}}
        result = d.get("Result")
        task_id = (result or {}).get("Id") if isinstance(result, dict) else None
        if task_id:
            print(f"\n=== SUCCESS task_id={task_id} ===")
        else:
            err = (d.get("ResponseMetadata", {}) or {}).get("Error")
            if err:
                print(f"\n=== API ERROR: {err} ===")
                sys.exit(1)
            print("\nNo task_id extracted — check raw response above")
    except Exception as e:
        print(f"\n=== ERROR: {type(e).__name__}: {e} ===")
        if hasattr(e, "body"):
            print(f"body: {e.body}")
        sys.exit(1)


if __name__ == "__main__":
    main()
