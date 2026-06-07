#!/usr/bin/env python3
"""临时 workaround: 拟合左臂关节零位 offset δq, 用修正后的 FK 重算左臂 T_base_ee,
拼一个临时 session 供 solve_calibration 重新求解。右臂/head 原样 symlink。

⚠️ 这是治标方案: 部署侧 FK 必须对左臂同样加 δq, 否则标定与部署不一致。
"""
import json
import os
import shutil
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, '/data1/tim/workspace/deepdive_kai0/calib')
import verify_projection as vp
from piper_fk import PiperFK

SRC = '/data1/tim/workspace/deepdive_kai0/calib/data/recalib'
DST = '/data1/tim/workspace/deepdive_kai0/calib/verify_out/offset_session'

fk = PiperFK()
calib = vp.load_calibration('/data1/tim/workspace/deepdive_kai0/calib/verify_out/recalib2_calibration.yml')
sess = vp.load_session(SRC)
L = [fr for fr in sess['frames'] if fr.arm == 'left']
Tlc = calib['transforms']['T_link6_camL']
Q = {fr.label: np.asarray(np.load(os.path.join(SRC, fr.label + '.npz'),
                                  allow_pickle=True)['joint_angles']) for fr in L}


def bb_resid(dq):
    ts = np.array([np.linalg.inv(fk.fk_homogeneous(Q[fr.label] + dq) @ Tlc @ fr.T_cam_board)[:3, 3]
                   for fr in L])
    return (ts - ts.mean(0)).ravel()


# 全量拟合 δq (bound ±4°)
res = least_squares(bb_resid, np.zeros(6), bounds=(-np.radians(4), np.radians(4)))
dq = res.x
print('拟合 δq (deg):', np.round(np.degrees(dq), 3).tolist())

# 构造临时 session: 左臂重存修正 npz, 右臂/head/pose_list symlink
if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(os.path.join(DST, 'left'))
os.makedirs(os.path.join(DST, 'right'))

for fr in L:
    d = dict(np.load(os.path.join(SRC, 'left', os.path.basename(fr.label) + '.npz'), allow_pickle=True))
    q = np.asarray(d['joint_angles'])
    d['T_base_ee'] = fk.fk_homogeneous(q + dq)        # 用修正 FK 覆盖
    np.savez(os.path.join(DST, 'left', os.path.basename(fr.label) + '.npz'), **d)
os.symlink(os.path.join(SRC, 'left', 'pose_list.json'), os.path.join(DST, 'left', 'pose_list.json'))

for fn in sorted(os.listdir(os.path.join(SRC, 'right'))):
    os.symlink(os.path.join(SRC, 'right', fn), os.path.join(DST, 'right', fn))
os.symlink(os.path.join(SRC, 'head.npz'), os.path.join(DST, 'head.npz'))
os.symlink(os.path.join(SRC, 'pose_list.json'), os.path.join(DST, 'pose_list.json'))

# 保存 δq 供部署侧使用
with open(os.path.join(os.path.dirname(DST), 'left_joint_offset_deg.json'), 'w') as f:
    json.dump({'arm': 'left', 'joint_offset_deg': np.degrees(dq).tolist(),
               'note': '部署侧左臂 FK 必须 q += radians(joint_offset_deg) 后再算 FK'}, f,
              ensure_ascii=False, indent=2)
print('临时 session 就绪:', DST)
print(f"左臂修正 npz: {len(L)} 帧; 右臂/head symlink 原始")
