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

  return (
    <div className="card controls-card">
      <h2>Recording {recording && <span style={{ color: "#f8514a" }}>● REC</span>}</h2>
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
