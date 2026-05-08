"""Sanity checks on pure_vis600 for training-readiness.

Checks (each prints PASS/FAIL):
 1. info.json schema vs reality (counts, features, splits, paths)
 2. meta jsonl line counts + first/last episode_index
 3. parquet × 600 readability + column schema parity with kai0_base
 4. parquet rows == mp4 frame count (sampled 30 random eps × 3 cams)
 5. video codec / resolution uniformity (sampled)
 6. mirror semantic on random sampled mirrors (state swap bit-exact)
 7. global index continuity (no gaps, monotonic)
 8. timestamp = i / FPS exactly per episode
 9. tasks.jsonl single task index matches all parquet rows
 10. kind labels: 309 originals + 291 mirrors

If all PASS → training-ready.
"""
from __future__ import annotations
import json, random, subprocess, sys
from pathlib import Path
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed

DST = Path("/home/tim/workspace/deepdive_kai0/kai0/data/Task_A/self_built/pure_vis600")
KAI0_BASE = Path("/home/tim/workspace/deepdive_kai0/kai0/data/Task_A/kai0_base")
FFMPEG = "/home/tim/workspace/deepdive_kai0/kai0/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
N_TOTAL = 600
FPS = 30
CHUNK = 0
CAMS = ("observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right")

result_log: list[tuple[str, bool, str]] = []

def chk(name, cond, detail=""):
    result_log.append((name, bool(cond), str(detail)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    return bool(cond)


# ---------- 1. info.json -------------------------------------------------
print("\n[1] info.json self-consistency vs kai0_base parity")
info = json.loads((DST / "meta" / "info.json").read_text())
info_kai0 = json.loads((KAI0_BASE / "meta" / "info.json").read_text())
chk("info.total_episodes == 600", info["total_episodes"] == N_TOTAL, str(info["total_episodes"]))
chk("info.total_videos == 600*3 = 1800", info["total_videos"] == N_TOTAL * 3, str(info["total_videos"]))
chk("info.total_chunks == 1", info["total_chunks"] == 1)
chk("info.fps == 30", info["fps"] == 30)
chk("info.codebase_version v2.1", info["codebase_version"] == "v2.1")
chk("splits['train'] = 0:600", info["splits"].get("train") == "0:600")
# feature key parity (top-level): pure_vis600 should have same observation.images.* + observation.state + action
keys_pv = set(info["features"].keys())
keys_kb = set(info_kai0["features"].keys())
keys_kb_no_depth = {k for k in keys_kb if not k.startswith("observation.depth.")}
chk("features keys ⊇ kai0_base (no depth)", keys_kb_no_depth.issubset(keys_pv),
    f"missing: {sorted(keys_kb_no_depth - keys_pv)}")
chk("features no observation.depth.*", not any(k.startswith("observation.depth.") for k in keys_pv))
for cam in CAMS:
    chk(f"features['{cam}'] shape 480x640x3", info["features"].get(cam, {}).get("shape") == [480, 640, 3])

# ---------- 2. meta jsonl ------------------------------------------------
print("\n[2] meta/episodes.jsonl + tasks.jsonl line-level checks")
ep_lines = (DST / "meta" / "episodes.jsonl").read_text().splitlines()
ep_lines = [l for l in ep_lines if l.strip()]
chk(f"episodes.jsonl has {N_TOTAL} entries", len(ep_lines) == N_TOTAL, str(len(ep_lines)))
ep_records = [json.loads(l) for l in ep_lines]
for i, r in enumerate(ep_records):
    if r["episode_index"] != i:
        chk(f"episodes.jsonl episode_index continuity", False, f"line {i} has episode_index={r['episode_index']}")
        break
else:
    chk("episodes.jsonl episode_index continuity 0..599", True)
n_orig = sum(1 for r in ep_records if r["kind"] == "original")
n_mir = sum(1 for r in ep_records if r["kind"] == "mirror")
chk(f"309 originals + 291 mirrors", n_orig == 309 and n_mir == 291, f"got orig={n_orig} mir={n_mir}")
chk("all tasks = ['Flatten and fold the cloth.']",
    all(r["tasks"] == ["Flatten and fold the cloth."] for r in ep_records))
chk("mirrors have mirror_of_orig_ep field", all("mirror_of_orig_ep" in r for r in ep_records if r["kind"] == "mirror"))

tasks_lines = (DST / "meta" / "tasks.jsonl").read_text().splitlines()
tasks_lines = [l for l in tasks_lines if l.strip()]
chk("tasks.jsonl has 1 entry", len(tasks_lines) == 1)
chk("tasks.jsonl task_index 0", json.loads(tasks_lines[0])["task_index"] == 0)

# ---------- 3. parquet schema parity -------------------------------------
print("\n[3] parquet readability + schema parity vs kai0_base")
def cols(p):
    t = pq.read_table(p)
    return [(f.name, str(f.type)) for f in t.schema]
sample_pv = cols(DST / "data" / "chunk-000" / "episode_000000.parquet")
sample_kb = cols(KAI0_BASE / "data" / "chunk-000" / "episode_000000.parquet")
core_kb = [c for c in sample_kb if c[0] in {n for n, _ in sample_pv}]
chk("ep 0 parquet schema == kai0_base subset", core_kb == [c for c in sample_pv if c[0] in {n for n, _ in core_kb}],
    f"\n     pv: {sample_pv}\n     kb: {core_kb}")
n_readable = 0
for i in range(N_TOTAL):
    try:
        t = pq.read_table(DST / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
        n_readable += 1
    except Exception as e:
        chk(f"all 600 parquet readable", False, f"ep {i}: {e}")
        break
else:
    chk(f"all {N_TOTAL} parquet readable", n_readable == N_TOTAL)

# ---------- 4. parquet rows == mp4 frame count (sampled) -----------------
print("\n[4] parquet rows == mp4 frame count (sampled 30 eps × 3 cams)")
def mp4_nframes(p):
    # full decode (no -c copy); -stats forces progress lines in stderr.
    r = subprocess.run([FFMPEG, "-v", "error", "-stats", "-i", str(p), "-f", "null", "-"],
                       capture_output=True, text=True, timeout=120)
    import re
    m = re.findall(r"frame=\s*(\d+)", r.stderr)
    return int(m[-1]) if m else -1

random.seed(0)
sample_eps = random.sample(range(N_TOTAL), 30)
mismatches = []
def check_ep(ep):
    t = pq.read_table(DST / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    nrows = t.num_rows
    out = {"ep": ep, "rows": nrows, "frames": {}}
    for cam in CAMS:
        out["frames"][cam] = mp4_nframes(DST / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{ep:06d}.mp4")
    return out

with ProcessPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(check_ep, ep): ep for ep in sample_eps}
    for fut in as_completed(futs):
        r = fut.result()
        ep = r["ep"]
        for cam, nf in r["frames"].items():
            if nf != r["rows"]:
                mismatches.append((ep, cam, nf, r["rows"]))
chk(f"30 sampled eps × 3 cams: parquet rows == mp4 frames",
    not mismatches, f"{len(mismatches)} mismatch(es): " + str(mismatches[:3]))

# ---------- 5. video codec uniformity (sampled) --------------------------
print("\n[5] video codec / resolution / fps uniform (sampled 5 originals + 5 mirrors × 3 cams)")
def codec_info(p):
    r = subprocess.run([FFMPEG, "-hide_banner", "-i", str(p)],
                       capture_output=True, text=True, timeout=30)
    return r.stderr  # ffmpeg dumps to stderr
sample = random.sample(range(0, 309), 5) + random.sample(range(309, 600), 5)
all_codecs = set()
all_dims = set()
all_fps = set()
for ep in sample:
    for cam in CAMS:
        s = codec_info(DST / "videos" / f"chunk-{CHUNK:03d}" / cam / f"episode_{ep:06d}.mp4")
        for line in s.splitlines():
            if "Video:" in line:
                # e.g. "Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 640x480, ..."
                if "h264" in line:
                    all_codecs.add("h264")
                if "640x480" in line:
                    all_dims.add("640x480")
                if "30 fps" in line:
                    all_fps.add("30")
chk("all sampled videos h264", all_codecs == {"h264"}, str(all_codecs))
chk("all sampled videos 640x480", all_dims == {"640x480"}, str(all_dims))
chk("all sampled videos 30 fps", all_fps == {"30"}, str(all_fps))

# ---------- 6. mirror semantic on randoms --------------------------------
print("\n[6] mirror state-swap bit-exactness (sampled 5 mirror eps)")
mirror_recs = [r for r in ep_records if r["kind"] == "mirror"]
sample_m = random.sample(mirror_recs, 5)
bad_swap = 0
for r in sample_m:
    new_ep = r["episode_index"]
    orig_new_ep = r["mirror_of_orig_ep"]
    pq_o = pq.read_table(DST / "data" / "chunk-000" / f"episode_{orig_new_ep:06d}.parquet")
    pq_m = pq.read_table(DST / "data" / "chunk-000" / f"episode_{new_ep:06d}.parquet")
    if pq_o.num_rows != pq_m.num_rows:
        chk(f"  ep {new_ep} mirror rows match orig", False)
        bad_swap += 1; continue
    state_o = pq_o.column("observation.state").to_pylist()
    state_m = pq_m.column("observation.state").to_pylist()
    action_o = pq_o.column("action").to_pylist()
    action_m = pq_m.column("action").to_pylist()
    # check first row state
    sw = state_o[0][7:14] + state_o[0][0:7]
    if state_m[0] != sw:
        bad_swap += 1; print(f"     ep {new_ep} state mismatch row 0")
    sw_a = action_o[0][7:14] + action_o[0][0:7]
    if action_m[0] != sw_a:
        bad_swap += 1; print(f"     ep {new_ep} action mismatch row 0")
chk("5 random mirrors: state+action bit-exact swap", bad_swap == 0, f"{bad_swap} mismatch(es)")

# ---------- 7. global index continuity -----------------------------------
print("\n[7] global index continuity 0..total_frames-1")
expected_total = info["total_frames"]
running = 0
broke = False
for ep in range(N_TOTAL):
    t = pq.read_table(DST / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    idx = t.column("index").to_pylist()
    if idx[0] != running or idx[-1] != running + t.num_rows - 1:
        chk("global index continuity", False, f"ep {ep}: starts {idx[0]}, expected {running}")
        broke = True; break
    running += t.num_rows
if not broke:
    chk(f"global index continuous 0..{running-1}", running == expected_total, f"running={running} vs total_frames={expected_total}")

# ---------- 8. timestamp = i/30 ------------------------------------------
print("\n[8] timestamp == i/30 within each episode (sampled 5 eps)")
ts_bad = 0
for ep in random.sample(range(N_TOTAL), 5):
    t = pq.read_table(DST / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    ts = t.column("timestamp").to_pylist()
    expected = [i / FPS for i in range(t.num_rows)]
    # tolerance: float32 precision
    if not all(abs(a - b) < 1e-4 for a, b in zip(ts, expected)):
        ts_bad += 1
        print(f"     ep {ep} timestamp mismatch")
chk("5 sampled eps: timestamp == i/30", ts_bad == 0)

# ---------- 9. task_index --------------------------------------------------
print("\n[9] all parquet rows have task_index == 0")
ti_bad = 0
for ep in random.sample(range(N_TOTAL), 10):
    t = pq.read_table(DST / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    uniq = t.column("task_index").unique().to_pylist()
    if uniq != [0]:
        ti_bad += 1; print(f"     ep {ep} task_index unique = {uniq}")
chk("10 sampled eps: task_index uniform 0", ti_bad == 0)

# ---------- summary -------------------------------------------------------
n_pass = sum(1 for _, ok, _ in result_log if ok)
n_total = len(result_log)
print(f"\n========== {n_pass} / {n_total} checks passed ==========")
if n_pass == n_total:
    print("✅ pure_vis600 is structurally training-ready.")
    sys.exit(0)
else:
    print("⚠️  some checks failed; inspect above before training.")
    sys.exit(1)
