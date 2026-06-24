import { useState } from "react";
import { api } from "../api";
import type { DaggerStatus } from "../types";

interface Props {
  s: DaggerStatus | null;
}

/** Recording controls — 开始 / 保存 / 丢弃, same logic as start_data_collect.sh:
 *   开始 start  : open a new dagger episode (HUMAN_RECORD, not yet recording)
 *   保存 save   : finalize + keep the current episode
 *   丢弃 discard: abort the current episode, delete partial files
 * The dagger episode writer is gated by these; the state machine
 * (POLICY_RUN ↔ ALIGNING ↔ HUMAN_RECORD ↔ RETURNING) is driven by the master
 * arm's freedrive switches (web read-only). The hardware F3 pedal remains a
 * start↔save toggle at the recorder level.
 */
export default function ControlsCard({ s }: Props) {
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const call = async (fn: () => Promise<unknown>) => {
    setErr(null); setBusy(true);
    try { await fn(); } catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };

  const rosAlive = !!s?.ros_alive;
  const sessionUp = !!s?.session_running;
  const inDagger = s?.state === "HUMAN_RECORD";
  const recording = !!s?.recording;

  // 开始 needs HUMAN_RECORD + not recording; 保存/丢弃 need an open episode.
  const canStart = rosAlive && sessionUp && inDagger && !recording;
  const canEnd = rosAlive && recording;
  // Per-rollout boundary: only meaningful while the policy is running (POLICY_RUN).
  const canRollout = rosAlive && sessionUp && s?.state === "POLICY_RUN";
  // rollout_paused: true = paused between rollouts (next press STARTS), false =
  // a rollout is running (next press ENDS), null = unknown.
  const paused = s?.rollout_paused;

  return (
    <div className="card controls-card">
      <h2>Rollout 采集</h2>
      {/* Current-state banner so the operator always knows what the button does next */}
      <div className="kv" style={{ marginBottom: 8 }}>
        <div className="k">当前</div>
        <div className="v">
          {!canRollout ? <span style={{ opacity: 0.6 }}>— (需 POLICY_RUN)</span>
            : paused === true ? <span style={{ color: "#d29922", fontWeight: 600 }}>⏸ 已暂停 · 重置场景后开始下一轮</span>
            : paused === false ? <span style={{ color: "#3fb950", fontWeight: 600 }}>● 采集中 · 本轮进行</span>
            : <span style={{ opacity: 0.6 }}>采集中 (状态未知)</span>}
        </div>
      </div>
      <div className="row-buttons">
        <button
          className={paused === true ? "" : "danger"}
          style={paused === true ? { background: "#238636", borderColor: "#2ea043", color: "white" } : undefined}
          disabled={!canRollout || busy}
          onClick={() => call(() => api.rolloutNext())}
          title="一轮 = 一次自主任务尝试(叠衣/抓取/擦拭…任意场景)。本轮完成按一次=结束并暂停(标 success);重置场景后再按一次=开始下一轮(自动 flush RTC, 模型不重载)">
          {paused === true ? "▶ 开始下一轮" : "⏹ 本轮完成 · 暂停"}
        </button>
      </div>
      <div className="hint">
        一轮 = 一次自主任务尝试 (不限叠衣, 抓取/擦拭等同理)。本轮完成按一次 = 切一个干净的 inference rollout(success) 并暂停;
        重置场景后再按一次 = 开始下一轮 (自动 flush RTC, 模型不重载)。失败由接管自动标记, 无需手动。
      </div>
      <h2 style={{ marginTop: 16 }}>Recording {recording && <span style={{ color: "#f8514a" }}>● REC</span>}</h2>
      <div className="row-buttons">
        <button className="primary" disabled={!canStart || busy}
                onClick={() => call(() => api.recordStart())}
                title={!inDagger ? "需进入 HUMAN_RECORD (拨开两个柔性开关)" :
                       recording ? "已在录制中" : "开始录制 dagger episode"}>
          ● 开始
        </button>
        <button disabled={!canEnd || busy}
                onClick={() => call(() => api.recordSave())}
                style={{ background: "#238636", borderColor: "#2ea043", color: "white" }}
                title="保存并结束当前 episode">
          ✓ 保存
        </button>
        <button className="danger" disabled={!canEnd || busy}
                onClick={() => call(() => api.recordDiscard())}
                title="丢弃当前 episode (删除半成品文件)">
          ✕ 丢弃
        </button>
      </div>
      <div className="hint">
        与 start_data_collect.sh 一致: 开始 → 保存 / 丢弃。仅在 HUMAN_RECORD
        (双柔性开关 ON) 有效, 不改变状态机。硬件 F3 踏板仍是 开始↔保存 切换。
      </div>
      <div className="kv" style={{ marginTop: 12 }}>
        <div className="k">Hardware pedal</div>
        <div className="v">
          {s?.last_pedal_ts
            ? <>fired {(performance.now() / 1000 - s.last_pedal_ts).toFixed(1)}s ago</>
            : "waiting…"}
        </div>
      </div>
      {err && <div className="error">{err}</div>}
    </div>
  );
}
