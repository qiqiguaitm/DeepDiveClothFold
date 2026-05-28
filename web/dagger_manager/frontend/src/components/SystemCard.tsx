import { useEffect, useState } from "react";
import { api } from "../api";
import type { CkptEntry, DaggerStatus } from "../types";

interface Props { s: DaggerStatus | null; }

// Supported ckpt directory groups. ckpt_v0 = JAX in-process; ckpt_v1 = V1
// Triton serve + websocket. Extend as new groups are packed (ckpt_v2, ...).
const ALLOWED_GROUPS = new Set<string>(["ckpt_v0", "ckpt_v1"]);

// A ckpt is launchable when its sidecar + variant-specific assets are present.
function ckptOk(c: CkptEntry): boolean {
  if (!c.has_sidecar) return false;
  if (c.variant === "v1") return c.has_v1_pkl && c.has_norm_stats;
  return c.has_norm_stats || !c.config_name;
}

export default function SystemCard({ s }: Props) {
  const [ckpts, setCkpts] = useState<CkptEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const reload = async () => {
    setErr(null);
    try {
      const all = await api.ckpts();
      setCkpts(all.filter(c => ALLOWED_GROUPS.has(c.group) && c.has_sidecar));
    } catch (e: any) { setErr(e?.message ?? String(e)); }
  };
  useEffect(() => { reload(); }, []);

  const selectedEntry = ckpts.find(c => c.path === selected) ?? null;
  const selectedValid = selectedEntry ? ckptOk(selectedEntry) : false;

  // While a session is running, lock the selection to the running ckpt.
  useEffect(() => {
    if (s?.session_running && s.ckpt) setSelected(s.ckpt);
  }, [s?.session_running, s?.ckpt]);

  // With the bundled lifecycle (start_dagger_collect.sh starts both infra +
  // web), the web cannot itself start/stop infra — the shell terminal owns
  // that. So "system up" here = session (policy_inference) running.
  // infraReady = dagger_recorder publishing /dagger/state, our proxy for
  // "cameras + arms + recorder are all alive".
  const infraReady = s?.state !== null && s?.state !== undefined;
  const sessionUp = !!s?.session_running;
  const systemUp = sessionUp;
  const starting = false;  // brief — system_start is async + short readiness wait

  const start = async () => {
    if (!selected || !selectedEntry) return;
    setErr(null); setBusy(true);
    try { await api.systemStart({ ckpt: selected, variant: selectedEntry.variant }); }
    catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };
  const stop = async () => {
    setErr(null); setBusy(true);
    try { await api.systemStop(); }
    catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };

  return (
    <div className="card ckpt-card">
      <h2>System</h2>
      <div className="kv" style={{ marginBottom: 12 }}>
        <div className="k">Infra</div>
        <div className="v" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className={`led ${infraReady ? "led-on" : "led-off"}`} />
          {infraReady ? "ready (shell-managed)" : "starting up…"}
        </div>
        <div className="k">Session</div>
        <div className="v" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span className={`led ${sessionUp ? "led-on" : "led-off"}`} />
          {sessionUp ? `loaded (pid ${s?.session_pid})` : "no policy loaded"}
          <div style={{ marginLeft: "auto" }}>
            {sessionUp ? (
              <button className="danger" onClick={stop} disabled={busy}>
                Stop
              </button>
            ) : (
              <button className="primary" onClick={start}
                      disabled={!selected || !selectedValid || !infraReady || busy}
                      title={!infraReady ? "infra not ready yet" :
                             !selected ? "select a ckpt below" :
                             !selectedValid ? "selected ckpt is missing required assets" :
                             selectedEntry?.variant === "v1"
                               ? "start V1 serve + websocket client (~30s)"
                               : "load JAX policy (~22s)"}>
                Start
              </button>
            )}
          </div>
        </div>
        <div className="k">Selected ckpt</div>
        <div className="v" style={{ fontFamily: "monospace", fontSize: 12 }}>
          {selectedEntry ? (
            <>
              <VariantBadge variant={selectedEntry.variant} /> {selected}
            </>
          ) : <span style={{ color: "#8b949e" }}>—</span>}
        </div>
      </div>

      <h2 style={{ marginTop: 4, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span>Checkpoint (ckpt_v0 / ckpt_v1)</span>
        <button onClick={reload} disabled={busy}
                style={{ fontSize: 11, padding: "3px 8px" }}>↻</button>
      </h2>
      <div className="ckpt-list">
        {ckpts.map((c) => {
          const ok = ckptOk(c);
          const locked = systemUp && selected !== c.path;
          return (
            <div
              key={c.path}
              className={`ckpt-row ${selected === c.path ? "selected" : ""}`}
              onClick={() => !systemUp && setSelected(c.path)}
              style={{ opacity: locked ? 0.4 : 1, cursor: systemUp ? "default" : "pointer" }}
            >
              <div>{ok ? "✓" : <span className="bad">!</span>}</div>
              <div>
                <div style={{ fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}>
                  <VariantBadge variant={c.variant} />
                  {c.name}
                </div>
                <div className="meta">
                  {c.config_name ?? "—"}
                  {c.task_hint && <> · {c.task_hint}</>}
                  {c.config_name && !c.has_norm_stats && <span className="bad"> · no norm_stats</span>}
                  {c.variant === "v1" && !c.has_v1_pkl && <span className="bad"> · no v1_p200.pkl</span>}
                </div>
              </div>
            </div>
          );
        })}
        {ckpts.length === 0 && <div className="hint">no ckpt_v0 / ckpt_v1 ckpts found</div>}
      </div>
      <div className="hint" style={{ marginTop: 8 }}>
        Infra (CAN/cameras/arms/dagger_recorder/pedal) is managed by the
        shell — Ctrl-C the start_dagger_collect.sh terminal to bring it
        down. Start loads the chosen ckpt: <b>v0</b> = JAX in-process (~22s);
        <b> v1</b> = V1 Triton serve + websocket client (~30s).
      </div>
      {err && <div className="error">{err}</div>}
    </div>
  );
}

function VariantBadge({ variant }: { variant: "v0" | "v1" }) {
  const isV1 = variant === "v1";
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, letterSpacing: "0.04em",
      padding: "1px 6px", borderRadius: 4,
      background: isV1 ? "#1f6feb33" : "#3fb95033",
      color: isV1 ? "#79c0ff" : "#3fb950",
      border: `1px solid ${isV1 ? "#1f6feb66" : "#3fb95066"}`,
      fontFamily: "ui-sans-serif, system-ui, sans-serif",
    }}>
      {variant.toUpperCase()}
    </span>
  );
}
