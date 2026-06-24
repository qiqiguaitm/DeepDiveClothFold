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

function stripVer(date: string): string {
  return date.replace(/-v\d+$/, "");  // 2026-06-15-v3 → 2026-06-15
}

export default function HistoryCard({ task, episodes, selected, onSelect, onReload }: Props) {
  const [filter, setFilter] = useState<"all" | "dagger" | "inference">("dagger");
  const [dateFilter, setDateFilter] = useState<string>("all");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const bySubset = episodes.filter(e => filter === "all" ? true : e.subset === filter);
  // Distinct dates (newest-first) available under the current subset filter.
  const dates = Array.from(new Set(bySubset.map(e => e.date))).sort().reverse();
  // If the selected date is no longer present (subset changed), fall back to "all".
  const activeDate = dateFilter !== "all" && dates.includes(dateFilter) ? dateFilter : "all";
  const shown = bySubset.filter(e => activeDate === "all" ? true : e.date === activeDate);

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
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
        <span style={{ fontSize: 11, opacity: 0.7 }}>日期</span>
        <select value={activeDate} onChange={(e) => setDateFilter(e.target.value)}
          style={{ fontSize: 11, padding: "2px 6px", flex: 1 }}>
          <option value="all">全部 ({bySubset.length})</option>
          {dates.map(d => (
            <option key={d} value={d}>
              {stripVer(d)} ({bySubset.filter(e => e.date === d).length})
            </option>
          ))}
        </select>
      </div>
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
                <span className="meta">{stripVer(e.date)}</span>
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
