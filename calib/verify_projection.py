#!/usr/bin/env python3
"""Offline projection verification for hand-eye + intrinsics calibration.

Reuses captured frames in data/calib_/ (no hardware). For each frame, projects
the known charuco board 3D corners back into the image via the full calibration
chain and compares against detected corners, then localizes error by layer.
"""
import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from board_def import BoardSpec, get_board
import solve_calibration  # for _robust_mean_se3
from scipy.spatial.transform import Rotation


@dataclass
class FrameData:
    """One captured pose. Pixel arrays are (N,2)/(N,) with matching charuco ids."""
    label: str                 # e.g. "left/pose_03", "head"
    arm: str | None            # "left" | "right" | None (head)
    rgb: np.ndarray            # (H,W,3) uint8 BGR
    K: np.ndarray              # (3,3)
    dist: np.ndarray           # (5,)
    corners_2d: np.ndarray     # (N,2) detected charuco corners
    ids: np.ndarray            # (N,) int charuco ids
    T_cam_board: np.ndarray    # (4,4) from per-frame solvePnP (rvec,tvec)
    T_base_ee: np.ndarray | None   # (4,4) FK, None for head
    pnp_err: float             # stored per-frame solvePnP residual (px)


def _rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """(rvec,tvec) -> 4x4 homogeneous T_cam_board."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def load_frame(npz_path: str, label: str, arm: str | None) -> FrameData:
    """Load one pose npz into a FrameData."""
    d = np.load(npz_path, allow_pickle=True)
    return FrameData(
        label=label,
        arm=arm,
        rgb=d["rgb_image"],
        K=d["camera_matrix"].astype(np.float64),
        dist=d["dist_coeffs"].astype(np.float64).reshape(-1),
        corners_2d=d["charuco_corners"].reshape(-1, 2).astype(np.float64),
        ids=d["charuco_ids"].reshape(-1).astype(int),
        T_cam_board=_rt_to_T(d["rvec"], d["tvec"]),
        T_base_ee=(d["T_base_ee"].astype(np.float64) if "T_base_ee" in d.files else None),
        pnp_err=float(np.asarray(d["reproj_err"]).reshape(-1)[0]),
    )


def load_session(session_dir: str) -> dict:
    """Load all left/right pose npz + head.npz from a calib session dir."""
    frames: list[FrameData] = []
    for arm in ("left", "right"):
        adir = os.path.join(session_dir, arm)
        if not os.path.isdir(adir):
            continue
        for fn in sorted(f for f in os.listdir(adir) if f.startswith("pose_") and f.endswith(".npz")):
            frames.append(load_frame(os.path.join(adir, fn), f"{arm}/{fn[:-4]}", arm))
    head = None
    head_path = os.path.join(session_dir, "head.npz")
    if os.path.exists(head_path):
        head = load_frame(head_path, "head", None)
    return {"frames": frames, "head": head}


def board_corners_3d(board: cv2.aruco.CharucoBoard, ids: np.ndarray) -> np.ndarray:
    """Charuco board interior corners (meters, board frame) for given ids -> (N,3)."""
    all_corners = np.array(board.getChessboardCorners(), dtype=np.float64)
    return all_corners[np.asarray(ids).reshape(-1)]


def arm_cam_pose_world(
    T_world_base: np.ndarray,
    T_base_ee: np.ndarray,
    T_link6_cam: np.ndarray,
) -> np.ndarray:
    """Full hand-eye chain -> camera pose in world (4x4).

    T_world_cam = T_world_base @ T_base_ee @ T_link6_cam
    """
    return T_world_base @ T_base_ee @ T_link6_cam


def project_world_to_pixels(
    P_world: np.ndarray,
    T_world_cam: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Project world 3D points into a camera image -> (N,2) pixels (with distortion).

    Args:
        P_world: (N,3) points in world frame.
        T_world_cam: (4,4) camera pose in world.
        K: (3,3) camera intrinsic matrix.
        dist: (5,) distortion coefficients.

    Returns:
        (N,2) pixel coordinates.
    """
    P_world = np.asarray(P_world, dtype=np.float64).reshape(-1, 3)
    T_cam_world = np.linalg.inv(T_world_cam)
    P_cam = (T_cam_world @ np.c_[P_world, np.ones(len(P_world))].T).T[:, :3]
    px, _ = cv2.projectPoints(P_cam, np.zeros(3), np.zeros(3), K, dist)
    return px.reshape(-1, 2)


def frame_T_world_board(
    fr: FrameData,
    T_world_base: np.ndarray,
    T_link6_cam: np.ndarray,
) -> np.ndarray:
    """反推该臂帧观测到的 board 世界位姿 T_world_board(i).

    Uses the arm hand-eye chain: T_world_cam = arm_cam_pose_world(...),
    then T_world_board = T_world_cam @ T_cam_board.
    """
    T_world_cam = arm_cam_pose_world(T_world_base, fr.T_base_ee, T_link6_cam)
    return T_world_cam @ fr.T_cam_board


def head_T_world_board(fr: FrameData, T_world_camF: np.ndarray) -> np.ndarray:
    """Head 帧观测到的 board 世界位姿.

    T_world_board = T_world_camF @ T_cam_board
    """
    return T_world_camF @ fr.T_cam_board


def se3_spread(T_list: list[np.ndarray]) -> tuple[float, float]:
    """Spread of a set of SE3 poses about their robust mean.

    Returns (translation_std_mm, rotation_std_deg).
    """
    T_mean = solve_calibration._robust_mean_se3(T_list)
    t_dev = [np.linalg.norm(T[:3, 3] - T_mean[:3, 3]) for T in T_list]
    R_mean_inv = T_mean[:3, :3].T
    r_dev = [np.linalg.norm(Rotation.from_matrix(R_mean_inv @ T[:3, :3]).as_rotvec())
             for T in T_list]
    return float(np.sqrt(np.mean(np.square(t_dev))) * 1000.0), \
        float(np.degrees(np.sqrt(np.mean(np.square(r_dev)))))


def load_calibration(path: str) -> dict:
    """Load calibration.yml transforms (list->ndarray) + intrinsics."""
    with open(path) as f:
        data = yaml.safe_load(f)
    for k in data["transforms"]:
        data["transforms"][k] = np.array(data["transforms"][k], dtype=np.float64)
    return data


def _arm_extrinsics(calib: dict, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_world_base, T_link6_cam) for the given arm."""
    t = calib["transforms"]
    if arm == "left":
        return t["T_world_baseL"], t["T_link6_camL"]
    return t["T_world_baseR"], t["T_link6_camR"]


def run_layers(session: dict, calib: dict, board: cv2.aruco.CharucoBoard) -> dict:
    """Run L0-L3 diagnostics. Returns a dict with layer results.

    Private keys prefixed with '_' store per-frame data for visualization;
    callers strip these before writing report.json.

    Args:
        session: Output of load_session().
        calib: Output of load_calibration().
        board: CharucoBoard instance from board_def.get_board().

    Returns:
        Dict with keys L0_intrinsics, L1_handeye, L2_reproj, L3_cross
        plus private _frames, _world_boards, _T_ref, _calib.
    """
    frames = session["frames"]
    head = session["head"]
    t = calib["transforms"]

    # --- Backproject T_world_board for each arm frame + collect camera poses ---
    world_boards: list[np.ndarray] = []
    per_frame: list[tuple] = []
    for fr in frames:
        T_world_base, T_link6_cam = _arm_extrinsics(calib, fr.arm)
        T_world_board_i = frame_T_world_board(fr, T_world_base, T_link6_cam)
        world_boards.append(T_world_board_i)
        per_frame.append((fr, arm_cam_pose_world(T_world_base, fr.T_base_ee, T_link6_cam)))
    head_world_board = head_T_world_board(head, t["T_world_camF"]) if head else None

    # Reference board pose: robust mean over all arm-frame backprojections
    T_ref = solve_calibration._robust_mean_se3(world_boards)

    # L0: intrinsics (PnP residual summary)
    pnp = np.array([fr.pnp_err for fr in frames] + ([head.pnp_err] if head else []))

    # L1: hand-eye self-consistency (spread of backprojected T_world_board)
    tr_mm, rot_deg = se3_spread(world_boards)

    # L2: end-to-end reprojection (project reference board into every frame)
    all_err: list[np.ndarray] = []
    frame_viz: list[dict] = []
    iter_list = list(per_frame) + ([(head, t["T_world_camF"])] if head else [])
    for fr, T_world_cam in iter_list:
        P_board = board_corners_3d(board, fr.ids)
        P_world = (T_ref @ np.c_[P_board, np.ones(len(P_board))].T).T[:, :3]
        pred = project_world_to_pixels(P_world, T_world_cam, fr.K, fr.dist)
        err = np.linalg.norm(pred - fr.corners_2d, axis=1)
        all_err.append(err)
        frame_viz.append({"label": fr.label, "det": fr.corners_2d, "pred": pred,
                          "err_mean": float(err.mean()), "rgb": fr.rgb})
    flat = np.concatenate(all_err)

    # L3: cross-camera / world-frame consistency
    posL = t["T_world_baseL"][:3, 3]
    posR = t["T_world_baseR"][:3, 3]
    sym_mm = float(np.linalg.norm((posL + posR) / 2.0) * 1000.0)
    head_arm_mm = (float(np.linalg.norm(head_world_board[:3, 3] - T_ref[:3, 3]) * 1000.0)
                   if head_world_board is not None else None)

    rep = {
        "session": None,  # caller (main) fills the real path
        "n_frames": len(frames) + (1 if head else 0),
        "L0_intrinsics": {"pnp_err_mean_px": float(pnp.mean()),
                          "pnp_err_max_px": float(pnp.max()),
                          "pass": bool(pnp.mean() < 0.5)},
        "L1_handeye": {"world_board_trans_std_mm": tr_mm,
                       "world_board_rot_std_deg": rot_deg,
                       "pass": bool(tr_mm < 3.0 and rot_deg < 0.3)},
        "L2_reproj": {"err_mean_px": float(flat.mean()),
                      "err_p95_px": float(np.percentile(flat, 95)),
                      "err_max_px": float(flat.max()),
                      "pass": bool(flat.mean() < 2.0 and np.percentile(flat, 95) < 5.0)},
        "L3_cross": {"base_symmetry_mm": sym_mm,
                     "head_vs_arm_board_mm": head_arm_mm,
                     "pass": bool(sym_mm < 5.0 and (head_arm_mm is None or head_arm_mm < 10.0))},
    }
    rep["_frames"] = frame_viz
    rep["_world_boards"] = world_boards
    rep["_T_ref"] = T_ref
    rep["_calib"] = calib
    return rep


def verdict(rep: dict) -> str:
    """One-line localization conclusion based on which layer first fails."""
    if not rep["L0_intrinsics"]["pass"]:
        return "内参/板检测异常 (L0): 重新标定内参"
    if not rep["L1_handeye"]["pass"]:
        return "外参不自洽 (L1): 嫌疑 hand-eye / FK / base-in-board"
    if not rep["L3_cross"]["pass"]:
        return "跨相机/世界系不一致 (L3): 嫌疑 T_world_camF / 世界系定义"
    if not rep["L2_reproj"]["pass"]:
        return "全链路重投影偏大 (L2) 但各层自洽: 嫌疑 FK/URDF 与部署不匹配"
    return "全部通过: 标定在采集数据上自洽且重投影良好"


def _overlay_png_b64(rgb: np.ndarray, det: np.ndarray, pred: np.ndarray) -> str:
    """Draw det(green)/pred(red)+连线 on RGB, return base64 PNG."""
    img = rgb.copy()
    for (du, dv), (pu, pv) in zip(det.astype(int), pred.astype(int)):
        cv2.line(img, (du, dv), (pu, pv), (0, 255, 255), 1)
        cv2.circle(img, (du, dv), 3, (0, 255, 0), -1)   # detected green
        cv2.circle(img, (pu, pv), 3, (0, 0, 255), -1)   # predicted red
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


def build_report_html(rep: dict, out_html: str) -> None:
    """Assemble the self-contained Plotly HTML report."""
    import plotly.graph_objects as go

    # 1) 3D world-frame consistency: board corners + base/cam poses
    board = get_board(BoardSpec.from_yaml(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_9x14.yaml")))
    all_corners = np.array(board.getChessboardCorners(), dtype=np.float64)
    fig3d = go.Figure()
    for Twb in rep["_world_boards"]:
        Pw = (Twb @ np.c_[all_corners, np.ones(len(all_corners))].T).T[:, :3]
        fig3d.add_trace(go.Scatter3d(x=Pw[:, 0], y=Pw[:, 1], z=Pw[:, 2],
                                     mode="markers", marker=dict(size=1.5),
                                     opacity=0.4, showlegend=False))
    t = rep["_calib"]["transforms"]
    for name, T in (("baseL", t["T_world_baseL"]), ("baseR", t["T_world_baseR"]),
                    ("camF", t["T_world_camF"])):
        p = T[:3, 3]
        fig3d.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text",
                                     marker=dict(size=5), text=[name], name=name))
    fig3d.update_layout(scene=dict(aspectmode="data"), title="世界系一致性 (board 角点应聚拢)")

    # 2) Per-frame error bar chart
    labels = [f["label"] for f in rep["_frames"]]
    errs = [f["err_mean"] for f in rep["_frames"]]
    figbar = go.Figure(go.Bar(x=labels, y=errs))
    figbar.update_layout(title="每帧全链路重投影误差 (px)", xaxis_tickangle=-60)

    # 3) Overlay images (base64 embedded)
    overlay_html = ""
    for f in rep["_frames"]:
        b64 = _overlay_png_b64(f["rgb"], f["det"], f["pred"])
        overlay_html += (f'<div style="display:inline-block;margin:4px;text-align:center">'
                         f'<img src="data:image/png;base64,{b64}" width="320"><br>'
                         f'<small>{f["label"]} — {f["err_mean"]:.2f}px</small></div>')

    head_html = (f'<h2>标定投影验证报告</h2><p><b>结论:</b> {verdict(rep)}</p>'
                 f'<pre>{json.dumps({k: v for k, v in rep.items() if not k.startswith("_")}, ensure_ascii=False, indent=2)}</pre>')

    with open(out_html, "w") as fp:
        fp.write("<html><head><meta charset='utf-8'></head><body>")
        fp.write(head_html)
        fp.write(fig3d.to_html(full_html=False, include_plotlyjs="cdn"))
        fp.write(figbar.to_html(full_html=False, include_plotlyjs=False))
        fp.write("<h3>叠加图 (绿=检测, 红=标定预测)</h3>")
        fp.write(overlay_html)
        fp.write("</body></html>")


def main() -> None:
    """CLI entry point for offline calibration projection verification."""
    ap = argparse.ArgumentParser(description="离线标定投影验证")
    ap.add_argument("--session", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "calib_"))
    ap.add_argument("--board", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "board_9x14.yaml"))
    args = ap.parse_args()

    board = get_board(BoardSpec.from_yaml(args.board))
    calib = load_calibration(os.path.join(args.session, "calibration.yml"))
    session = load_session(args.session)
    rep = run_layers(session, calib, board)

    out_dir = os.path.join(args.session, "verify")
    os.makedirs(out_dir, exist_ok=True)
    clean = {k: v for k, v in rep.items() if not k.startswith("_")}
    clean["session"] = os.path.abspath(args.session)
    clean["verdict"] = verdict(rep)
    with open(os.path.join(out_dir, "report.json"), "w") as fp:
        json.dump(clean, fp, ensure_ascii=False, indent=2)
    build_report_html(rep, os.path.join(out_dir, "verify_report.html"))

    print("=" * 60)
    print(f"结论: {verdict(rep)}")
    for k in ("L0_intrinsics", "L1_handeye", "L2_reproj", "L3_cross"):
        print(f"  {k}: {'PASS' if rep[k]['pass'] else 'FAIL'}  {clean[k]}")
    print(f"报告: {os.path.join(out_dir, 'verify_report.html')}")


if __name__ == "__main__":
    main()
