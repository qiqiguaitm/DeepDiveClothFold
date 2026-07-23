#!/usr/bin/env python3
"""停止火山 ML Platform 任务(自定义任务 = API 里的 job)。

需要 env: VOLC_AK / VOLC_SK。用法:
    VOLC_AK=... VOLC_SK=... python volc_job_stop.py t-2026xxxx-abcde [t-...]
与 volc_job_status.py 同鉴权模式(volcengine.base.Service 直调 OpenAPI, 绕 SDK 反序列化 bug)。
"""
import json, os, sys
from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service


def get_svc(region="cn-beijing"):
    si = ServiceInfo('open.volcengineapi.com', {'Accept': 'application/json'},
                     Credentials(os.environ['VOLC_AK'], os.environ['VOLC_SK'], 'ml_platform', region), 5, 5)
    return Service(si, {
        'StopJob': ApiInfo('POST', '/', {'Action': 'StopJob', 'Version': '2024-07-01'}, {}, {}),
        'GetJob':  ApiInfo('POST', '/', {'Action': 'GetJob',  'Version': '2024-07-01'}, {}, {}),
    })


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("用法: python volc_job_stop.py <task_id> [task_id...]")
    svc = get_svc()
    for jid in sys.argv[1:]:
        try:
            r = svc.json('StopJob', {}, json.dumps({"Id": jid}).encode())
            print(f"{jid}: STOP 已发出 -> {json.loads(r).get('Result', {})}")
        except Exception as e:
            print(f"{jid}: STOP 失败 {e}")
        try:
            g = json.loads(svc.json('GetJob', {}, json.dumps({"Id": jid}).encode()))
            print(f"   当前状态: {g['Result']['Status']['State']}")
        except Exception as e:
            print(f"   查状态失败 {e}")
