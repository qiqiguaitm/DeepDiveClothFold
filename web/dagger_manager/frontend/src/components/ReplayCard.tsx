import { useRef, useState } from "react";
import { api } from "../api";
import type { DaggerStatus, EpisodeEntry } from "../types";

interface Props {
  s: DaggerStatus | null;
  ep: EpisodeEntry | null;
  task: string;
  onDeleted: () => void;
}

const CAMS = ["hand_left", "top_head", "hand_right"] as const;

export default function ReplayCard({ ep, task, onDeleted }: Props) {
  const refs = useRef<Record<string, HTMLVideoElement | null>>({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const playAll = () => Object.values(refs.current).forEach(v => v && v.play());
  const pauseAll = () => Object.values(refs.current).forEach(v => v && v.pause());
  const seekAll = (t: number) =>
    Object.values(refs.current).forEach(v => { if (v) v.currentTime = t; });

  const del = async () => {
    if (!ep) return;
    if (!confirm(`删除 ${task} ${ep.subset}/${ep.date.replace(/-v\d+$/, "")} #${ep.episode_id}?\n` +
                 `parquet + 视频 + meta 都会删除, 不可恢复。`)) return;
    setBusy(true); setErr(null);
    try {
      await api.delEpisode(ep.subset, ep.date, ep.episode_id, task);
      onDeleted();
    } catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };

  if (!ep) {
    return (
      <div className="card replay-card">
        <h2>Replay</h2>
        <div className="hint">从 History 选一条 episode 回放。</div>
      </div>
    );
  }

  return (
    <div className="card replay-card">
      <h2>
        Replay — {ep.subset} #{ep.episode_id} · {ep.date.replace(/-v\d+$/, "")} ·
        {" "}{ep.duration_s.toFixed(1)}s · {ep.length}f
      </h2>
      <div style={{ color: "#8b949e", fontSize: 12, marginBottom: 8 }}>
        prompt: {ep.prompt || "—"}{ep.note ? ` · ${ep.note}` : ""}
      </div>
      {ep.has_video ? (
        <>
          <div className="replay-vids">
            {CAMS.map(cam => (
              <video key={`${ep.subset}/${ep.date}/${ep.episode_id}/${cam}/${ep.created_at}`}
                ref={el => (refs.current[cam] = el)}
                src={api.episodeVideoUrl(ep.subset, ep.date, ep.episode_id, cam, task, ep.created_at)}
                controls={false} muted preload="metadata"
                style={{ width: "100%", background: "#000", borderRadius: 4 }} />
            ))}
          </div>
          <div className="row-buttons" style={{ marginTop: 8 }}>
            <button onClick={playAll} className="primary">▶ Play</button>
            <button onClick={pauseAll}>⏸ Pause</button>
            <button onClick={() => seekAll(0)}>⏮ Restart</button>
            <button onClick={del} className="danger" disabled={busy}
                    style={{ marginLeft: "auto" }}>🗑 删除此 episode</button>
          </div>
          {err && <div className="error">{err}</div>}
          <div className="hint" style={{ marginTop: 6 }}>
            AV1 → H.264 transcode on the fly (server). 真机 kinematic replay 仍走
            CLI: <code>start_scripts/kai/start_replay_test.sh</code>.
          </div>
        </>
      ) : (
        <>
          <div className="error">该 episode 没有视频文件（可能录制中断）。</div>
          <div className="row-buttons" style={{ marginTop: 8 }}>
            <button onClick={del} className="danger" disabled={busy}>🗑 删除此 episode</button>
          </div>
          {err && <div className="error">{err}</div>}
        </>
      )}
    </div>
  );
}
