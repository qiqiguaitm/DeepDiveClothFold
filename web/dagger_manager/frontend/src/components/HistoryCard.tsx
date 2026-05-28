import { useState } from "react";
import { api } from "../api";
import type { EpisodeEntry } from "../types";

interface Props {
  task: string;
  episodes: EpisodeEntry[];
  selected: EpisodeEntry | null;
  onSelect: (e: EpisodeEntry | null) => void;
  onReload: () => void;
}

function epKey(e: EpisodeEntry): string {
  return `${e.subset}/${e.date}/${e.episode_id}`;
}

export default function HistoryCard({ task, episodes, selected, onSelect, onReload }: Props) {
  const [filter, setFilter] = useState<"all" | "dagger" | "inference">("dagger");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const shown = episodes.filter(e => filter === "all" ? true : e.subset === filter);

  const del = async (e: EpisodeEntry) => {
    if (!confirm(`删除 ${task} ${epKey(e)}? 不可恢复。`)) return;
    setBusy(true); setErr(null);
    try {
      await api.delEpisode(e.subset, e.date, e.episode_id, task);
      if (selected && epKey(selected) === epKey(e)) onSelect(null);
      onReload();
    } catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };

  return (
    <div className="card history-card">
      <h2 style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span>History · {task} ({shown.length})</span>
        <span style={{ display: "flex", gap: 4 }}>
          {(["dagger", "inference", "all"] as const).map(f => (
            <button key={f}
              onClick={() => setFilter(f)}
              className={filter === f ? "primary" : ""}
              style={{ fontSize: 11, padding: "2px 8px" }}>
              {f}
            </button>
          ))}
          <button onClick={onReload} disabled={busy} style={{ fontSize: 11, padding: "2px 8px" }}>↻</button>
        </span>
      </h2>
      <div className="ep-list">
        {shown.map((e) => {
          const sel = selected && epKey(selected) === epKey(e);
          return (
            <div key={epKey(e)}
              className={`ep-row ${sel ? "selected" : ""}`}
              onClick={() => onSelect(e)}>
              <div className="ep-main">
                <span className={`ep-tag ep-${e.subset}`}>{e.subset === "dagger" ? "D" : "I"}</span>
                <span style={{ fontWeight: 500 }}>#{e.episode_id}</span>
                <span className="meta">{e.date.replace("-v2", "")}</span>
              </div>
              <div className="ep-stats">
                <span>{e.length}f · {e.duration_s.toFixed(1)}s</span>
                {!e.has_video && <span className="bad"> · no video</span>}
                <button className="ep-del" disabled={busy}
                  onClick={(ev) => { ev.stopPropagation(); del(e); }}
                  title="delete">✕</button>
              </div>
            </div>
          );
        })}
        {shown.length === 0 && <div className="hint">no episodes</div>}
      </div>
      {err && <div className="error">{err}</div>}
    </div>
  );
}
