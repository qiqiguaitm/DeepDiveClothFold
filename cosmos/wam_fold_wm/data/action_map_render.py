"""EVAC-style spatial action-map renderer for visrobot01 (faithful port of EnerVerse-AC's
`ddpm3d.py::get_traj`, adapted to our data: 14-dim joint state → FK → world EEF → project → render).

Reference: /mnt/pfs/p46h4f/cosmos/deepdive_kai0/enerverse-ac/lvdm/models/ddpm3d.py::get_traj
           + lvdm/data/traj_vis_statistics.py (constants).

Pipeline (per camera):
  state[T,14] = [L 6 joints, L gripper, R 6 joints, R gripper]
  -> PiperFK(6 joints) -> T_base_ee  (per arm)
  -> world EEF: T_world_ee = T_world_base · T_base_ee
  -> pose[T,16] = [L xyz(3), L quat xyzw(4), L grip(1), R xyz(3), R quat(4), R grip(1)]
  -> EVAC get_traj: project EndEffectorPts via (w2c · pose · Gripper2EEFCvt) then K;
     filled circle at EEF origin (color = gripper open/close), 3 axis lines (orientation).
Output: action map [T, H, W, 3] uint8 (per camera). Downsample to latent res for cond_tokens.
"""
import os
import sys
import numpy as np
import yaml
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.cm as cm

_REPO = "/mnt/pfs/p46h4f/cosmos/deepdive_kai0"
sys.path.insert(0, os.path.join(_REPO, "calib"))
from piper_ik import PiperIK  # noqa: E402  (ikpy/URDF FK, self-contained — no piper_sdk needed)

# --- EVAC constants (verbatim from traj_vis_statistics.py) ---
ColorMapLeft = cm.Greens
ColorMapRight = cm.Reds
ColorListLeft = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
ColorListRight = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]
EndEffectorPts = np.array([[0, 0, 0, 1], [0.1, 0, 0, 1], [0, 0.1, 0, 1], [0, 0, 0.1, 1]], dtype=np.float64)
Gripper2EEFCvt = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.23], [0, 0, 0, 1]], dtype=np.float64)
_GRIPPER_NORM = 0.08  # visrobot01 gripper opening range ~0..0.07 m (NOT AgiBot's 120); color norm only


def _T(mat):
    return np.array(mat, dtype=np.float64).reshape(4, 4)


class ActionMapRenderer:
    """Render EVAC spatial action maps for one camera of the visrobot01 rig."""

    def __init__(self, calib_yml=None, cam="cam_f", grip_norm=_GRIPPER_NORM, radius=12):
        calib_yml = calib_yml or os.path.join(_REPO, "config/calibration.yml")
        c = yaml.safe_load(open(calib_yml))
        tr, intr = c["transforms"], c["intrinsics"]
        self.grip_norm = grip_norm
        self.radius = radius
        self.cam = cam
        # intrinsics K (3x3)
        ci = intr[cam]
        self.W, self.H = int(ci["width"]), int(ci["height"])
        cx = ci.get("cx", self.W / 2.0)
        cy = ci.get("cy", self.H / 2.0)
        self.K = np.array([[ci["fx"], 0, cx], [0, ci["fy"], cy], [0, 0, 1]], dtype=np.float64)
        # world->camera (w2c). calibration stores T_world_cam (cam->world); w2c = inv.
        cam_key = {"cam_f": "T_world_camF", "cam_l": "T_link6_camL", "cam_r": "T_link6_camR"}[cam]
        if cam == "cam_f":
            self.w2c = np.linalg.inv(_T(tr["T_world_camF"]))
            self.wrist = None
        else:
            # wrist cams move with the arm: c2w(t) = T_world_base · T_base_ee(t) · T_link6_cam
            base = "T_world_baseL" if cam == "cam_l" else "T_world_baseR"
            self.wrist = (cam[-1], _T(tr[base]), _T(tr[cam_key]))  # (arm, T_world_base, T_link6_cam)
        self.T_world_baseL = _T(tr["T_world_baseL"])
        self.T_world_baseR = _T(tr["T_world_baseR"])
        self.fk = PiperIK()

    def _world_ee(self, q6, T_world_base):
        T = T_world_base @ self.fk.fk_homogeneous(np.asarray(q6, dtype=np.float64))
        xyz = T[:3, 3]
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        return xyz, quat, T

    def state_to_pose16(self, state):
        """state[T,14] -> pose[T,16] world-frame [L xyz,quat,grip, R xyz,quat,grip]."""
        state = np.asarray(state, dtype=np.float64)
        T = state.shape[0]
        pose = np.zeros((T, 16), dtype=np.float64)
        for i in range(T):
            lx, lq, _ = self._world_ee(state[i, 0:6], self.T_world_baseL)
            rx, rq, _ = self._world_ee(state[i, 7:13], self.T_world_baseR)
            pose[i, 0:3], pose[i, 3:7], pose[i, 7] = lx, lq, state[i, 6]
            pose[i, 8:11], pose[i, 11:15], pose[i, 15] = rx, rq, state[i, 13]
        return pose

    @staticmethod
    def _mat_from_quat(p7):  # [x,y,z, qx,qy,qz,qw] -> 4x4
        M = np.eye(4)
        M[:3, :3] = Rotation.from_quat(p7[3:7]).as_matrix()
        M[:3, 3] = p7[0:3]
        return M

    def render(self, state, w2c_per_frame=None):
        """state[T,14] -> action_map [T,H,W,3] uint8. For wrist cams pass per-frame w2c."""
        pose = self.state_to_pose16(state)
        Tn = pose.shape[0]
        imgs = np.zeros((Tn, self.H, self.W, 3), dtype=np.uint8) + 50
        for i in range(Tn):
            w2c = w2c_per_frame[i] if w2c_per_frame is not None else self.w2c
            for side, p_off, gv, cmap, axc in (
                ("L", 0, pose[i, 7], ColorMapLeft, ColorListLeft),
                ("R", 8, pose[i, 15], ColorMapRight, ColorListRight),
            ):
                ee_mat = self._mat_from_quat(pose[i, p_off:p_off + 7])
                ee2cam = w2c @ ee_mat @ Gripper2EEFCvt
                pts = (ee2cam @ EndEffectorPts.T)  # [4coords, 4pts]
                uvw = self.K @ pts[:3, :]
                z = pts[2:3, :]
                z[z == 0] = 1e-6
                uv = (uvw / z)[:2, :].T.astype(np.int64)  # [4pts, 2]
                base = uv[0]
                if base[0] < 0 or base[0] >= self.W or base[1] < 0 or base[1] >= self.H:
                    continue
                col = tuple(int(c * 255) for c in cmap(float(gv) / self.grip_norm)[:3])
                cv2.circle(imgs[i], (int(base[0]), int(base[1])), self.radius, col, -1)
                for k in (1, 2, 3):
                    cv2.line(imgs[i], (int(base[0]), int(base[1])), (int(uv[k][0]), int(uv[k][1])), axc[k - 1], 4)
        return imgs


class MultiViewActionMapRenderer:
    """Render the 3-camera STACKED action map matching the WM's stacked video layout
    (wam_fold_dataset: top row = head/cam_f full-size; bottom row = [left|right] wrist views each
    resized to (H//2, W//2) and concatenated). Output [T, H+H//2, W, 3] uint8.

    Head uses a static w2c; wrist cams use per-frame poses w2c(t)=inv(T_world_base·FK(joints)·T_link6_cam).
    Both arms' world-frame EEFs are projected into each view.
    """

    def __init__(self, calib_yml=None, grip_norm=_GRIPPER_NORM, radius=12):
        self.f = ActionMapRenderer(calib_yml, cam="cam_f", grip_norm=grip_norm, radius=radius)
        self.l = ActionMapRenderer(calib_yml, cam="cam_l", grip_norm=grip_norm, radius=max(6, radius // 2))
        self.r = ActionMapRenderer(calib_yml, cam="cam_r", grip_norm=grip_norm, radius=max(6, radius // 2))

    def _wrist_w2c(self, ren, state):
        """Per-frame world->camera for a wrist cam: inv(T_world_base · FK(arm joints) · T_link6_cam)."""
        arm, T_world_base, T_link6_cam = ren.wrist  # arm in {'l','r'}
        q_slice = slice(0, 6) if arm == "l" else slice(7, 13)
        out = []
        for i in range(state.shape[0]):
            T_base_ee = ren.fk.fk_homogeneous(np.asarray(state[i, q_slice], dtype=np.float64))
            c2w = T_world_base @ T_base_ee @ T_link6_cam
            out.append(np.linalg.inv(c2w))
        return out

    def render(self, state):
        state = np.asarray(state, dtype=np.float64)
        head = self.f.render(state)  # [T,H,W,3]
        left = self.l.render(state, w2c_per_frame=self._wrist_w2c(self.l, state))
        right = self.r.render(state, w2c_per_frame=self._wrist_w2c(self.r, state))
        T, H, W, _ = head.shape
        hh, hw = H // 2, W // 2
        out = np.zeros((T, H + hh, W, 3), dtype=np.uint8)
        for t in range(T):
            out[t, :H] = head[t]
            lo = cv2.resize(left[t], (hw, hh), interpolation=cv2.INTER_AREA)
            ro = cv2.resize(right[t], (hw, hh), interpolation=cv2.INTER_AREA)
            out[t, H:] = np.concatenate([lo, ro], axis=1)
        return out  # [T, H+H//2, W, 3]


if __name__ == "__main__":
    # Self-test: render a sample visrobot01 episode's action map for top_head and overlay on the
    # real frame to verify the EEF projection lands on the gripper.
    import pandas as pd
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default="visrobot01_v3_train")
    ap.add_argument("--cam", default="cam_f")
    ap.add_argument("--grip-norm", type=float, default=120.0)
    ap.add_argument("--out", default="/tmp/claude-0/-mnt-pfs-p46h4f-cosmos-deepdive-kai0-cosmos/81fc32bb-d091-4b0e-967f-1d4657f90b8e/scratchpad/actmap_test.png")
    a = ap.parse_args()
    root = f"{_REPO}/kai0/data/wam_fold_v3/{a.rig}"
    pq = sorted(__import__("glob").glob(f"{root}/data/chunk-000/episode_*.parquet"))[0]
    df = pd.read_parquet(pq)
    state = np.stack(df["observation.state"].values)[:30]  # [30,14]
    r = ActionMapRenderer(cam=a.cam, grip_norm=a.grip_norm)
    amap = r.render(state)
    print(f"[actmap] rendered {amap.shape} for {a.rig}/{a.cam}; nonzero px frame0: {(amap[0] != 50).any(-1).sum()}")
    cv2.imwrite(a.out, amap[0][:, :, ::-1])
    print(f"[actmap] saved frame0 -> {a.out}")
