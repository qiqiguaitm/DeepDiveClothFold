"""P0 read-only probe: verify the joint-space (action_in_dim=14) reinit plan.

Confirms that constructing tau0's WanModel with action_in_dim=14 keeps EVERY
pretrained-trunk parameter name (video `blocks`, `action_blocks`, embeddings,
heads) aligned with the released checkpoint, and that the ONLY parameters that
become "new" are the two action-space-bound projections:
  - action_proj_in  (Linear: 20->1024  =>  14->1024)
  - action_head     (Head out: 20      =>  14)

Name-level check needs only diffusion_pytorch_model.bin.index.json (already on
disk). If both .bin shards are fully present it additionally does a real
strict=False load and reports the empirical missing / unexpected / shape-mismatch.

Read-only: never writes weights.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tau0_wm repo root
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from models.wan_2_2_models.transformers.model import WanModel  # noqa: E402

CKPT_DIR = os.path.join(ROOT, "checkpoints", "tau-0-wm")
INDEX = os.path.join(CKPT_DIR, "diffusion_pytorch_model.bin.index.json")

# Diffusion config from configs/deployment/wan_pretrain_rela_eef6d.yaml,
# with action_in_dim overridden 20 -> 14 (joint space).
CFG = dict(
    model_type="ti2v",
    patch_size=[1, 2, 2],
    text_len=512,
    in_dim=48,
    dim=3072,
    ffn_dim=14336,
    freq_dim=256,
    text_dim=4096,
    out_dim=48,
    num_heads=24,
    num_layers=30,
    window_size=[-1, 1],
    qk_norm=True,
    cross_attn_norm=True,
    eps=1.0e-06,
    use_ae=True,
    action_in_dim=14,   # <-- joint space (released ckpt is 20 = eef6d)
    action_dim=1024,
    action_num_heads=16,
    action_ffn_dim=2048,
    action_max_seq_len=60,
)
RELEASED_ACTION_IN_DIM = 20


def build_model_meta():
    """Instantiate on meta device if accelerate is available (no 20GB alloc)."""
    try:
        from accelerate import init_empty_weights
        with init_empty_weights():
            return WanModel(**CFG), "meta"
    except Exception as e:
        print(f"[info] init_empty_weights unavailable ({e}); building on CPU (957GB RAM ok)")
        return WanModel(**CFG), "cpu"


def trunk_family(k):
    p = k.split(".")
    if p[0] == "blocks":
        return "blocks (video backbone)"
    if p[0] == "action_blocks":
        return "action_blocks (pretrained action trunk)"
    if p[0] in ("action_proj_in", "action_head"):
        return "** action projection (NEW / joint) **"
    return "other trunk (embeddings/heads/freqs)"


def main():
    assert os.path.exists(INDEX), f"missing index: {INDEX}"
    ckpt_keys = set(json.load(open(INDEX))["weight_map"].keys())

    model, dev = build_model_meta()
    msd = model.state_dict()
    model_keys = set(msd.keys())

    # buffers like self.freqs / action_freqs are not in the index (registered as plain
    # attributes, recomputed at init) — exclude pure-buffer names from the name diff.
    param_names = {n for n, _ in model.named_parameters()}

    missing = sorted((param_names - ckpt_keys))           # model params absent from ckpt
    unexpected = sorted((ckpt_keys - model_keys))          # ckpt keys with no home in model

    print("=" * 78)
    print(f"P0 PROBE  (model on {dev}; action_in_dim={CFG['action_in_dim']}, released={RELEASED_ACTION_IN_DIM})")
    print("=" * 78)
    print(f"ckpt param keys (index.json): {len(ckpt_keys)}")
    print(f"model param keys             : {len(param_names)}")

    # coverage by family
    import collections
    cov = collections.Counter(trunk_family(k) for k in (param_names & ckpt_keys))
    print("\n-- name-matched params by family (these load from pretrained) --")
    for k, v in sorted(cov.items()):
        print(f"  {v:5d}  {k}")

    print(f"\n-- model params MISSING from ckpt (would be NEW / random-init): {len(missing)} --")
    for k in missing:
        shp = tuple(msd[k].shape)
        print(f"   {k:40s} model-shape={shp}")

    print(f"\n-- ckpt keys UNEXPECTED by model: {len(unexpected)} --")
    for k in unexpected:
        print(f"   {k}")

    # The two action-space-bound projections: present by NAME but shape differs (14 vs 20).
    print("\n-- action-space-bound projections (present by name; shape changes 20->14) --")
    for k in sorted(param_names):
        if k.startswith("action_proj_in") or k.startswith("action_head"):
            print(f"   {k:40s} model-shape={tuple(msd[k].shape)}  in_ckpt={k in ckpt_keys}")

    # Verdict
    only_proj_changes = all(
        m.startswith("action_proj_in") or m.startswith("action_head") for m in missing
    )
    print("\n" + "=" * 78)
    if not unexpected and (not missing or only_proj_changes):
        print("VERDICT: ✅ trunk fully aligned by name; only action_proj_in/action_head are")
        print("         action-space-bound (shape 20->14). Reuse plan root CONFIRMED.")
    else:
        print("VERDICT: ⚠️  unexpected name diffs — inspect above (naming may differ).")
    print("=" * 78)

    # Optional: empirical load if both shards complete.
    sizes = {"diffusion_pytorch_model-00001-of-00002.bin": 9999883021,
             "diffusion_pytorch_model-00002-of-00002.bin": 1023548491}
    have_shards = all(
        os.path.exists(os.path.join(CKPT_DIR, f)) and os.path.getsize(os.path.join(CKPT_DIR, f)) == sz
        for f, sz in sizes.items()
    )
    if not have_shards:
        print("\n[info] shards not both complete yet -> skipped empirical shape/load check.")
        print("       re-run after download to get real missing/mismatched from load_state_dict.")
        return

    print("\n[info] both shards complete -> running real strict=False load for shape check ...")
    from utils.model_utils import load_index_file
    sd = load_index_file(INDEX, mode="bin")
    if dev == "meta":
        model, _ = WanModel(**CFG), "cpu"  # need real params to load into
        model = WanModel(**CFG)
    mismatch = []
    msd = model.state_dict()
    for k in list(sd.keys()):
        if k in msd and sd[k].shape != msd[k].shape:
            mismatch.append((k, tuple(sd[k].shape), tuple(msd[k].shape)))
            del sd[k]
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  shape-mismatch (ckpt vs model, dropped): {len(mismatch)}")
    for k, a, b in mismatch:
        print(f"     {k:40s} ckpt={a}  model={b}")
    print(f"  missing after load (new params): {len(miss)} -> {list(miss)[:8]}{' ...' if len(miss)>8 else ''}")
    print(f"  unexpected: {len(unexp)} -> {list(unexp)[:8]}{' ...' if len(unexp)>8 else ''}")


if __name__ == "__main__":
    main()
