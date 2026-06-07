#!/usr/bin/env python3
"""用外参把每帧检测到的 board 角点融合到世界系, 看是否聚成一块平整板。
对比 offset 标定 (左臂 FK 加 δq) vs 无 offset (recalib2)。
理想: 同一 board 角点被所有帧投到世界系同一位置 (聚成一点)。
"""
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, '/data1/tim/workspace/deepdive_kai0/calib')
import verify_projection as vp
from board_def import BoardSpec, get_board
from piper_fk import PiperFK

CAL = '/data1/tim/workspace/deepdive_kai0/calib'
SRC = os.path.join(CAL, 'data/recalib')
fk = PiperFK()
board = get_board(BoardSpec.from_yaml(os.path.join(CAL, 'board_9x14.yaml')))
dq = np.radians(json.load(open(os.path.join(CAL, 'verify_out/left_joint_offset_deg.json')))['joint_offset_deg'])
Q = {}


def q_of(label):
    if label not in Q:
        Q[label] = np.asarray(np.load(os.path.join(SRC, label + '.npz'), allow_pickle=True)['joint_angles'])
    return Q[label]


def frame_world_board(calib, fr, use_offset):
    """该帧 board 角点在世界系 -> (N,3), 及 ids。"""
    if fr.arm is None:                                   # head
        Twb = calib['transforms']['T_world_camF'] @ fr.T_cam_board
    else:
        Tb, Tc = vp._arm_extrinsics(calib, fr.arm)
        if fr.arm == 'left' and use_offset:
            Tbe = fk.fk_homogeneous(q_of(fr.label) + dq)
        else:
            Tbe = fr.T_base_ee
        Twb = Tb @ Tbe @ Tc @ fr.T_cam_board
    P = vp.board_corners_3d(board, fr.ids)
    Pw = (Twb @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    return Pw, fr.ids


def collect(calib, use_offset):
    sess = vp.load_session(SRC)
    frames = sess['frames'] + [sess['head']]
    pts = {'head': [], 'left': [], 'right': []}
    byid = {}
    for fr in frames:
        Pw, ids = frame_world_board(calib, fr, use_offset)
        pts['head' if fr.arm is None else fr.arm].append(Pw)
        for p, cid in zip(Pw, ids):
            byid.setdefault(int(cid), []).append(p)
    pts = {k: (np.vstack(v) if v else np.empty((0, 3))) for k, v in pts.items()}
    # tightness: 每个角点 id 的世界位置散布 (mm)
    stds = [np.sqrt(((np.array(v) - np.array(v).mean(0)) ** 2).sum(1).mean())
            for v in byid.values() if len(v) > 1]
    return pts, float(np.mean(stds) * 1000)


COLOR = {'head': '#e41a1c', 'left': '#4daf4a', 'right': '#377eb8'}
calib_off = vp.load_calibration(os.path.join(CAL, 'verify_out/calibration_offset.yml'))
calib_no = vp.load_calibration(os.path.join(CAL, 'verify_out/recalib2_calibration.yml'))
pts_off, tight_off = collect(calib_off, True)
pts_no, tight_no = collect(calib_no, False)

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                    subplot_titles=(f'无 offset (recalib2): 角点散布 {tight_no:.1f}mm',
                                    f'offset 标定: 角点散布 {tight_off:.1f}mm'))
for col, pts in ((1, pts_no), (2, pts_off)):
    for cam, P in pts.items():
        if len(P) == 0:
            continue
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode='markers',
                                   marker=dict(size=1.6, color=COLOR[cam]), name=cam,
                                   legendgroup=cam, showlegend=(col == 1)), row=1, col=col)
for s in ('scene', 'scene2'):
    fig.layout[s].aspectmode = 'data'
fig.update_layout(title='board 角点融合到世界系 (越聚拢=外参越好; 绿=左臂 蓝=右臂 红=head)',
                  height=700)
out = os.path.join(CAL, 'verify_out/fuse_world.html')
fig.write_html(out, include_plotlyjs='cdn')
print('无 offset 角点散布: %.1f mm | offset: %.1f mm' % (tight_no, tight_off))
print('saved:', out)
