"""Convert GigaWorld per-embodiment joint norm-stats -> tau0 statistics.json format.

GigaWorld already computed joint-14 normalization on this exact data
(assets_visrobot01/norm_stats_{vis,kai}.json), in its delta convention
(arm joints as delta, gripper absolute). tau0's TauPolicy reads
statistics.json as {"action":{mean,std}, "state":{mean,std}} -> direct map.

Reusing these guarantees consistency with the action convention the deployment
server (inference_server.py add_state_to_action mask=delta_mask) already uses.

NOTE on convention: action.mean/std here are stats of the DELTA action for the
delta dims. The trainer/dataloader MUST apply the matching delta transform
(action_delta = action - state for delta_mask dims; gripper kept absolute)
before normalizing. delta_mask (idx 6,13 = grippers = absolute):
  [1,1,1,1,1,1,0, 1,1,1,1,1,1,0]
"""
import json
import os

GWP = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0/giga_world_policy/assets_visrobot01"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
DELTA_MASK = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0]  # 0 = gripper = absolute


def convert(src_name, dst_name):
    d = json.load(open(os.path.join(GWP, src_name)))["norm_stats"]
    out = {
        "action": {"mean": d["action"]["mean"], "std": d["action"]["std"]},
        "state": {"mean": d["observation.state"]["mean"], "std": d["observation.state"]["std"]},
        "_meta": {
            "space": "joint-14",
            "delta_mask": DELTA_MASK,
            "source": f"giga_world_policy/assets_visrobot01/{src_name}",
            "note": "action stats are in delta convention for delta_mask dims (gripper absolute)",
        },
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, dst_name)
    json.dump(out, open(path, "w"), indent=2)
    assert len(out["action"]["mean"]) == 14 and len(out["state"]["std"]) == 14
    print(f"wrote {path}")
    print(f"   action.mean[:4]={['%.3f'%x for x in out['action']['mean'][:4]]}  "
          f"state.std[:4]={['%.3f'%x for x in out['state']['std'][:4]]}")


if __name__ == "__main__":
    convert("norm_stats_vis.json", "statistics_visrobot01.json")  # embed_id 0 (target)
    convert("norm_stats_kai.json", "statistics_kairobot01.json")  # embed_id 1 (aux)
    print("done.")
