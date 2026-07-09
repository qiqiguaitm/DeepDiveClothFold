#!/usr/bin/env python3
"""Tiny launcher: runs train_lmwm2 and writes both stdout+stderr to a log file from within Python."""
import sys, subprocess, os
from pathlib import Path

code_dim = sys.argv[1]; tag = sys.argv[2]
REPO = Path("/home/tim/workspace/deepdive_kai0")
log = REPO / f"lmwm/outputs/lmwm2/{tag}.log"
log.parent.mkdir(parents=True, exist_ok=True)

with open(log, "a") as f:
    f.write(f"START {tag} code_dim={code_dim}\n"); f.flush()
    p = subprocess.run(
        [str(REPO / "kai0/.venv/bin/python"), "-u",
         str(REPO / "lmwm/scripts/train_lmwm2.py"),
         "--datasets", "kai0,coffee,xvla", "--cond", "prevz",
         "--steps", "6000", "--code_dim", code_dim, "--tag", tag],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(REPO))
    for line in p.stdout.splitlines():
        f.write(line + "\n")
    f.write(f"DONE {tag} rc={p.returncode}\n")
print(f"DONE {tag} rc={p.returncode} log={log}")
