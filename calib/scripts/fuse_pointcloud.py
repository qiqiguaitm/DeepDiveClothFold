#!/usr/bin/env python3
"""head + 左臂 + 右臂 各一帧, 用 offset 外参把 depth 反投影成带色点云融合到世界系。

近似说明: 用 depth 内参反投影得 depth 相机系点云, 用 color 内参取色、用 color 外参
变换 (忽略 depth-color 基线)。左右臂 D405 depth≈color, 准; head D435 depth/color
差异大, 有 cm 级偏移。
"""
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, '/data1/tim/workspace/deepdive_kai0/calib')
import verify_projection as vp
from piper_fk import PiperFK

CAL = '/data1/tim/workspace/deepdive_kai0/calib'
SRC = os.path.join(CAL, 'data/recalib')
fk = PiperFK()
calib = vp.load_calibration(os.path.join(CAL, 'verify_out/calibration_offset.yml'))
dq = np.radians(json.load(open(os.path.join(CAL, 'verify_out/left_joint_offset_deg.json')))['joint_offset_deg'])

# 各挑一帧 (近距离、板清晰)
FRAMES = [('head', 'head.npz', None), ('left', 'left/pose_05.npz', 'left'), ('right', 'right/pose_05.npz', 'right')]


def cam_pose_world(arm, npz):
    if arm is None:
        return calib['transforms']['T_world_camF']
    Tb, Tc = vp._arm_extrinsics(calib, arm)
    q = np.asarray(npz['joint_angles'])
    Tbe = fk.fk_homogeneous(q + dq) if arm == 'left' else fk.fk_homogeneous(q)
    return Tb @ Tbe @ Tc


def backproject(npz, T_world_cam, step=4, zmax=1.2):
    """depth → 带色世界点云。返回 (P_world Nx3, colors 'rgb()' list)。"""
    depth = npz['depth_image'].astype(np.float32)
    ds = float(np.asanyarray(npz['depth_scale']).reshape(-1)[0])
    di = json.loads(str(npz['depth_intrinsics']))
    K = npz['camera_matrix']
    rgb = npz['rgb_image']                       # BGR (cv2)
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:step, 0:w:step]
    Z = depth[::step, ::step] * ds
    m = (Z > 0.05) & (Z < zmax)
    uu, vv, Z = uu[m], vv[m], Z[m]
    # depth 相机系点云
    X = (uu - di['cx']) / di['fx'] * Z
    Y = (vv - di['cy']) / di['fy'] * Z
    P = np.stack([X, Y, Z], 1)
    # 取色: 用 color 内参把点投到 color 图 (近似同光心)
    uc = np.round(K[0, 0] * X / Z + K[0, 2]).astype(int)
    vc = np.round(K[1, 1] * Y / Z + K[1, 2]).astype(int)
    ok = (uc >= 0) & (uc < w) & (vc >= 0) & (vc < h)
    P = P[ok]
    bgr = rgb[vc[ok], uc[ok]]
    cols = ['rgb(%d,%d,%d)' % (c[2], c[1], c[0]) for c in bgr]   # BGR→RGB
    # 变换到世界系
    Pw = (T_world_cam @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    return Pw, cols


fig = go.Figure()
for name, rel, arm in FRAMES:
    npz = np.load(os.path.join(SRC, rel), allow_pickle=True)
    Twc = cam_pose_world(arm, npz)
    Pw, cols = backproject(npz, Twc)
    fig.add_trace(go.Scatter3d(x=Pw[:, 0], y=Pw[:, 1], z=Pw[:, 2], mode='markers',
                               marker=dict(size=1.4, color=cols), name=name))
    print('%-5s %d 点' % (name, len(Pw)))

# 画三个相机光心
for name, rel, arm in FRAMES:
    npz = np.load(os.path.join(SRC, rel), allow_pickle=True)
    p = cam_pose_world(arm, npz)[:3, 3]
    fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode='markers+text',
                               marker=dict(size=6, color='black'), text=[name + '_cam'],
                               showlegend=False))
fig.update_layout(scene=dict(aspectmode='data'),
                  title='depth 点云反投影融合到世界系 (offset 外参; head 因 depth-color 未对齐有 cm 级偏移)',
                  height=750)
out = os.path.join(CAL, 'verify_out/fuse_pointcloud.html')
fig.write_html(out, include_plotlyjs='cdn')
print('saved:', out)
