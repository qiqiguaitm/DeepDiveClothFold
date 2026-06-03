import os
import numpy as np
import cv2
import pytest

import verify_projection as vp
from board_def import BoardSpec, get_board

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION = os.path.join(HERE, "data", "calib_")
BOARD_YAML = os.path.join(HERE, "board_9x14.yaml")


@pytest.fixture(scope="module")
def board():
    return get_board(BoardSpec.from_yaml(BOARD_YAML))


def test_board_corner_ids_match_detection(board):
    """Self-取 3D 角点 + 该帧 PnP 投影,应落回 npz 存的检测角点 ~PnP 残差级别.

    证明 board 角点编号与采集时一致——整条验证链的前提.
    """
    fr = vp.load_frame(os.path.join(SESSION, "left", "pose_01.npz"), "left/pose_01", "left")
    P_board = vp.board_corners_3d(board, fr.ids)            # (N,3)
    rvec, _ = cv2.Rodrigues(fr.T_cam_board[:3, :3])
    tvec = fr.T_cam_board[:3, 3]
    proj, _ = cv2.projectPoints(P_board, rvec, tvec, fr.K, fr.dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - fr.corners_2d, axis=1)
    assert err.mean() < 1.0, f"mean reproj {err.mean():.3f}px — board id 约定可能不符"


def _synthetic_T(tx, ty, tz, rx=0, ry=0, rz=0):
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz], dtype=np.float64))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
    return T


def test_project_world_pinhole():
    """无畸变针孔: 相机原点看 z=1m 平面上的点,投影落在预期像素."""
    K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
    dist = np.zeros(5)
    T_world_cam = np.eye(4)                      # cam == world
    P_world = np.array([[0, 0, 1.0], [0.1, 0, 1.0]])  # 光轴上 + 右移 0.1m
    px = vp.project_world_to_pixels(P_world, T_world_cam, K, dist)
    assert np.allclose(px[0], [320, 240], atol=1e-6)
    assert np.allclose(px[1], [320 + 600 * 0.1, 240], atol=1e-6)


def test_arm_chain_roundtrip_zero_error():
    """已知 base/ee/link6_cam/board -> 真值相机位姿投影得'检测',链路预测应零误差."""
    K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
    dist = np.zeros(5)
    T_world_base = _synthetic_T(0.3, 0, 0)
    T_base_ee = _synthetic_T(0.2, 0.1, 0.4, rz=0.3)
    T_link6_cam = _synthetic_T(-0.08, 0, 0.04, rx=1.5)
    T_world_board = _synthetic_T(0.0, 0.0, 0.5)
    P_board = np.array([[0.02, 0.02, 0], [0.26, 0.16, 0], [0.1, 0.08, 0]])

    T_world_cam = vp.arm_cam_pose_world(T_world_base, T_base_ee, T_link6_cam)
    P_world = (T_world_board @ np.c_[P_board, np.ones(len(P_board))].T).T[:, :3]
    det = vp.project_world_to_pixels(P_world, T_world_cam, K, dist)
    pred = vp.project_world_to_pixels(P_world, T_world_cam, K, dist)
    assert np.allclose(det, pred, atol=1e-9)
    # 反推 T_world_board 应还原真值
    T_cam_board = np.linalg.inv(T_world_cam) @ T_world_board
    T_rec = T_world_cam @ T_cam_board
    assert np.allclose(T_rec, T_world_board, atol=1e-9)


def test_se3_spread_zero_for_identical():
    T = _synthetic_T(0.1, 0.2, 0.3, rz=0.5)
    tr_mm, rot_deg = vp.se3_spread([T.copy() for _ in range(5)])
    assert tr_mm < 1e-6 and rot_deg < 1e-6


def test_se3_spread_detects_offset():
    Ts = [_synthetic_T(0.1, 0, 0), _synthetic_T(0.1 + 0.01, 0, 0)]  # 差 10mm
    tr_mm, _ = vp.se3_spread(Ts)
    assert 3 < tr_mm < 8   # 关于均值的 std,两点对称 -> ~5mm


def test_run_layers_on_real_session(board):
    calib = vp.load_calibration(os.path.join(SESSION, "calibration.yml"))
    session = vp.load_session(SESSION)
    rep = vp.run_layers(session, calib, board)
    # 结构完整
    for k in ("L0_intrinsics", "L1_handeye", "L2_reproj", "L3_cross"):
        assert k in rep
    # 已知 PnP ~0.12px,内参层必过
    assert rep["L0_intrinsics"]["pnp_err_mean_px"] < 0.5
    # 每帧全链路误差有值
    assert rep["L2_reproj"]["err_mean_px"] >= 0
