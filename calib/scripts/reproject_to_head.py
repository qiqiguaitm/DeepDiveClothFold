#!/usr/bin/env python3
"""跨相机重投影验证: 左/右臂各一帧检测到的 board 角点, 经世界系(offset 外参)
重投影到 head 相机图像, 画在 head 的 RGB 上。

判读: board 固定 → 左右臂看到的是同一块板。绿=head 自己检测的角点(真值),
红=左臂经世界系投来, 蓝=右臂经世界系投来。红蓝落在绿附近 = 跨相机外参一致。
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, '/data1/tim/workspace/deepdive_kai0/calib')
import verify_projection as vp
from board_def import BoardSpec, get_board
from piper_fk import PiperFK

CAL = '/data1/tim/workspace/deepdive_kai0/calib'
SRC = os.path.join(CAL, 'data/recalib')
fk = PiperFK()
board = get_board(BoardSpec.from_yaml(os.path.join(CAL, 'board_9x14.yaml')))
calib = vp.load_calibration(os.path.join(CAL, 'verify_out/calibration_offset.yml'))
dq = np.radians(json.load(open(os.path.join(CAL, 'verify_out/left_joint_offset_deg.json')))['joint_offset_deg'])

# head 帧
head = vp.load_frame(os.path.join(SRC, 'head.npz'), 'head', None)
K_h = head.K
dist_h = head.dist
T_world_camF = calib['transforms']['T_world_camF']
T_cam_world = np.linalg.inv(T_world_camF)
img = head.rgb.copy()


def arm_board_in_world(rel, arm):
    """某臂帧检测到的 board 角点 → 世界系 (N,3), 用 offset 外参。"""
    fr = vp.load_frame(os.path.join(SRC, rel), rel, arm)
    Tb, Tc = vp._arm_extrinsics(calib, arm)
    q = np.asarray(np.load(os.path.join(SRC, rel), allow_pickle=True)['joint_angles'])
    Tbe = fk.fk_homogeneous(q + dq) if arm == 'left' else fk.fk_homogeneous(q)
    T_world_board = Tb @ Tbe @ Tc @ fr.T_cam_board
    P = vp.board_corners_3d(board, fr.ids)
    return (T_world_board @ np.c_[P, np.ones(len(P))].T).T[:, :3]


def draw_proj(P_world, color, label):
    """世界点投到 head 图并画点。"""
    P_cam = (T_cam_world @ np.c_[P_world, np.ones(len(P_world))].T).T[:, :3]
    front = P_cam[:, 2] > 0
    px, _ = cv2.projectPoints(P_cam[front], np.zeros(3), np.zeros(3), K_h, dist_h)
    px = px.reshape(-1, 2)
    for u, v in px.astype(int):
        cv2.circle(img, (u, v), 2, color, -1)
    return px


# 绿: head 自己检测的角点 (真值参照, 空心十字便于看清对齐)
for u, v in head.corners_2d.astype(int):
    cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 8, 1)
# 红: 左臂; 蓝: 右臂 (经世界系投来)
pxL = draw_proj(arm_board_in_world('left/pose_05.npz', 'left'), (0, 0, 255), 'left')
pxR = draw_proj(arm_board_in_world('right/pose_05.npz', 'right'), (255, 0, 0), 'right')

# 图例
cv2.putText(img, 'green=head detected  red=left->world->head  blue=right->world->head',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
cv2.putText(img, 'green=head  red=left  blue=right',
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

out = os.path.join(CAL, 'verify_out/reproject_to_head.png')
cv2.imwrite(out, img)
print('head 检测角点 %d, 左臂投影 %d, 右臂投影 %d' % (len(head.corners_2d), len(pxL), len(pxR)))
print('saved:', out)
