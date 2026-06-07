"""Orchestrate the 3-model eval across local GPUs.

For each model: split worklist into shards, launch one worker per GPU-group
(Nano: 1 GPU/worker; 64B: 2 GPU/worker, sharded transformer), wait, move on.
Results -> results/<model>_<shard>.jsonl ; videos -> report_assets/.
"""
import os, sys, json, subprocess, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/mnt/pfs/p46h4f/cosmos/cosmos3-venv/bin/python"
MROOT = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/cosmos/models/modelscope"
MODELS = {  # name -> (dir, mode, gpus_per_worker)
    "Cosmos3-Nano": (f"{MROOT}/Cosmos3-Nano", "single", 1),
    "Cosmos3-Super": (f"{MROOT}/Cosmos3-Super", "shard", 2),
    "Cosmos3-Super-Image2Video": (f"{MROOT}/Cosmos3-Super-Image2Video", "shard", 2),
}

def shard(units, k):
    return [units[i::k] for i in range(k)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--worklist", default=f"{HERE}/worklist.json")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    gpus = args.gpus.split(",")
    units = json.load(open(args.worklist))
    os.makedirs(f"{HERE}/results", exist_ok=True)
    os.makedirs(f"{HERE}/shards", exist_ok=True)
    os.makedirs(f"{HERE}/report_assets", exist_ok=True)

    for mname in args.models:
        mdir, mode, gpw = MODELS[mname]
        groups = [gpus[i:i + gpw] for i in range(0, len(gpus) - gpw + 1, gpw)]
        parts = shard(units, len(groups))
        print(f"\n=== {mname} ({mode}) : {len(groups)} workers x {gpw} GPU, {len(units)} units ===", flush=True)
        procs = []
        for wi, (grp, part) in enumerate(zip(groups, parts)):
            wf = f"{HERE}/shards/{mname}_{wi}.json"; json.dump(part, open(wf, "w"))
            out = f"{HERE}/results/{mname}_{wi}.jsonl"
            log = f"{HERE}/results/{mname}_{wi}.log"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(grp))
            cmd = [PY, f"{HERE}/eval_cosmos3.py", "--model", mdir, "--mode", mode,
                   "--work", wf, "--out", out, "--report_dir", f"{HERE}/report_assets", "--steps", str(args.steps)]
            procs.append(subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT))
            print(f"  worker {wi} GPU={grp} units={len(part)} -> {os.path.basename(out)}", flush=True)
        t0 = time.time()
        for p in procs: p.wait()
        print(f"=== {mname} done in {(time.time()-t0)/60:.1f} min ; merging+reporting ===", flush=True)
        # incremental: merge this model's window clips, regenerate all reports
        subprocess.run([PY, f"{HERE}/merge_videos.py", "--model", mname])
        subprocess.run([PY, f"{HERE}/gen_reports.py"])
    print("=== ALL MODELS DONE ===", flush=True)
    subprocess.run([PY, f"{HERE}/gen_reports.py"])

if __name__ == "__main__":
    main()
