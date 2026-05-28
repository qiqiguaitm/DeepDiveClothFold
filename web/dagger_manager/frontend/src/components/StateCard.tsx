import type { DaggerStatus } from "../types";

function recentMs(ts: number | null | undefined): string {
  if (!ts) return "—";
  // Note: server reports monotonic ts; we don't know browser monotonic so
  // we just compare to "now" wall as a rough freshness indicator.
  const age = Math.max(0, performance.now() / 1000 - ts);
  if (age > 60) return ">60s ago";
  return `${age.toFixed(1)}s ago`;
}

export default function StateCard({ s }: { s: DaggerStatus | null }) {
  const state = s?.state ?? "unknown";
  const cls = state === "unknown" ? "state-unknown" : `state-${state}`;
  const rec = !!s?.recording;
  return (
    <div className="card state-card">
      <h2>State</h2>
      <div style={{ marginBottom: 12, display: "flex", alignItems: "center", gap: 12 }}>
        <span className={`state-badge ${cls}`}>{state}</span>
        {rec && (
          <span style={{ color: "#f8514a", fontWeight: 600 }}>
            <span className="rec-dot" />REC
          </span>
        )}
      </div>
      <div className="kv">
        <div className="k">Stack</div>
        <div className="v">
          {s?.stack_running ? (
            <><span className="led led-on" />running (pid {s.stack_pid})</>
          ) : (
            <><span className="led led-off" />stopped</>
          )}
        </div>
        <div className="k">ROS bridge</div>
        <div className="v">{s?.ros_alive ? "alive" : "down"}</div>
        <div className="k">policy execute</div>
        <div className="v">
          {s?.policy_execute === null || s?.policy_execute === undefined
            ? "—"
            : s.policy_execute ? "enabled" : "halted"}
        </div>
        <div className="k">Button L / R</div>
        <div className="v">
          <span className={`led ${s?.button_left ? "led-on" : "led-off"}`} />L
          <span style={{ marginLeft: 14 }} />
          <span className={`led ${s?.button_right ? "led-on" : "led-off"}`} />R
        </div>
        <div className="k">last pedal</div>
        <div className="v">{recentMs(s?.last_pedal_ts ?? null)}</div>
      </div>
    </div>
  );
}
