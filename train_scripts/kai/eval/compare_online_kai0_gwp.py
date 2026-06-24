#!/usr/bin/env python3
"""真机在线 shadow 对比:kai0 执行折叠,gwp 影子(同一帧预测、不执行),逐帧记录动作分歧。

设计 (模式 a):
  - kai0 正常跑并 execute(start_autonomy_from_ckpt_v1.sh ... --execute) → 真叠衣服,
    其 raw 预测块发在 /policy/action_chunk(本脚本只读,不给 kai0 加负载)。
  - gwp ws server 单独起(serve_gwp_ws.py --port 8003),**没有节点驱动它**(纯影子)。
  - 本节点订阅相机+关节,组 gwp obs,query gwp:8003 拿 [48,14];取最近的 kai0 块;
    逐帧算 MAE@{1,8,16} + 两者 action_motion + gwp state |z| → csv + 控制台。

判读:
  - gwp 影子块与 kai0 块差很大(MAE 大) → 即使在 kai0 走出的好状态上,gwp 预测也偏 → 开环/接口问题。
  - 两者接近但 gwp 自己 execute 会垮 → 闭环 exposure bias。
  - gwp_motion ≈ 0 → gwp 塌缩成静止(对应真机"停顿")。

运行 (ROS 已 source; kai0/.venv 有 rclpy+websockets+openpi_client):
  source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
  kai0/.venv/bin/python train_scripts/kai/eval/compare_online_kai0_gwp.py \
      --gwp-port 8003 --rate 4 --out /data2/gwp_eval/out/online_shadow_compare.csv
"""
import argparse, csv, threading, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray
from openpi_client import websocket_client_policy as wcp

HOR = [1, 8, 16]
PROMPT = "Flatten and fold the cloth."


def _img(msg: Image) -> np.ndarray:
    # rgb8 -> HWC uint8 (gwp server _chw01 兼容 HWC/uint8)
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()


class ShadowCompare(Node):
    def __init__(self, args):
        super().__init__("compare_online_kai0_gwp")
        self.args = args
        self.lock = threading.Lock()
        self._img = {"top_head": None, "hand_left": None, "hand_right": None}
        self._jl = None; self._jr = None
        self._kai0_chunk = None
        self.rows = []

        self.create_subscription(Image, "/camera_f/camera/color/image_raw", lambda m: self._cbi(m, "top_head"), 1)
        self.create_subscription(Image, "/camera_l/camera/color/image_raw", lambda m: self._cbi(m, "hand_left"), 1)
        self.create_subscription(Image, "/camera_r/camera/color/image_raw", lambda m: self._cbi(m, "hand_right"), 1)
        self.create_subscription(JointState, "/puppet/joint_left", self._cbl, 10)
        self.create_subscription(JointState, "/puppet/joint_right", self._cbr, 10)
        self.create_subscription(Float32MultiArray, "/policy/action_chunk", self._cbk, 5)

        self.get_logger().info(f"connecting gwp ws server 127.0.0.1:{args.gwp_port} ...")
        self.gwp = wcp.WebsocketClientPolicy(host="127.0.0.1", port=args.gwp_port)
        self.get_logger().info(f"gwp metadata: {self.gwp.get_server_metadata()}")
        self.create_timer(1.0 / args.rate, self._tick)
        self.get_logger().info(f"shadow compare up @ {args.rate}Hz -> {args.out}")

    def _cbi(self, m, k):
        with self.lock: self._img[k] = _img(m)
    def _cbl(self, m):
        with self.lock: self._jl = np.asarray(m.position[:7], np.float32)
    def _cbr(self, m):
        with self.lock: self._jr = np.asarray(m.position[:7], np.float32)
    def _cbk(self, m):
        n = m.layout.dim[1].size if len(m.layout.dim) >= 2 else 14
        with self.lock: self._kai0_chunk = np.asarray(m.data, np.float32).reshape(-1, n)[:, :14]

    def _tick(self):
        with self.lock:
            if any(self._img[k] is None for k in self._img) or self._jl is None or self._jr is None:
                return
            obs = {"state": np.concatenate([self._jl, self._jr]).astype(np.float32),
                   "images": {k: self._img[k].copy() for k in self._img}, "prompt": PROMPT}
            kai0 = None if self._kai0_chunk is None else self._kai0_chunk.copy()
        try:
            gwp = np.asarray(self.gwp.infer(obs)["actions"], np.float32)[:, :14]
        except Exception as e:
            self.get_logger().warn(f"gwp infer failed: {e}"); return

        gm = float(np.abs(np.diff(gwp, axis=0)).mean())
        row = {"t": round(time.time(), 3), "gwp_motion": round(gm, 4)}
        if kai0 is not None and len(kai0) > 0:
            L = min(len(kai0), len(gwp))
            ae = np.abs(kai0[:L] - gwp[:L])
            for h in HOR:
                row[f"mae@{h}"] = round(float(ae[h - 1].mean()), 4) if h <= L else None
            row["mae_all"] = round(float(ae.mean()), 4)
            row["kai0_motion"] = round(float(np.abs(np.diff(kai0, axis=0)).mean()), 4)
        else:
            for h in HOR: row[f"mae@{h}"] = None
            row["mae_all"] = None; row["kai0_motion"] = None
        self.rows.append(row)
        self.get_logger().info(
            f"mae@1={row['mae@1']} @8={row['mae@8']} @16={row['mae@16']} all={row['mae_all']} "
            f"| gwp_motion={row['gwp_motion']} kai0_motion={row['kai0_motion']}")

    def dump(self):
        if not self.rows: return
        cols = ["t", "mae@1", "mae@8", "mae@16", "mae_all", "gwp_motion", "kai0_motion"]
        with open(self.args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in self.rows: w.writerow({c: r.get(c) for c in cols})
        valid = [r for r in self.rows if r["mae_all"] is not None]
        if valid:
            import statistics as st
            print(f"\n=== shadow summary ({len(valid)} frames w/ kai0 chunk) ===")
            for c in ("mae@1", "mae@8", "mae@16", "mae_all", "gwp_motion", "kai0_motion"):
                vals = [r[c] for r in valid if r.get(c) is not None]
                if vals: print(f"  {c:12s} mean={st.mean(vals):.4f}  max={max(vals):.4f}")
        print(f"saved -> {self.args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gwp-port", type=int, default=8003)
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--out", default="/data2/gwp_eval/out/online_shadow_compare.csv")
    args = ap.parse_args()
    rclpy.init()
    node = ShadowCompare(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.dump(); node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
