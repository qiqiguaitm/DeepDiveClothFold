import { useCallback, useEffect, useRef, useState } from "react";
import { api, connectStatusWs } from "./api";
import ArmsPanel from "./components/ArmsPanel";
import CameraGrid from "./components/CameraGrid";
import ControlsCard from "./components/ControlsCard";
import EpisodesCard from "./components/EpisodesCard";
import HistoryCard from "./components/HistoryCard";
import ReplayCard from "./components/ReplayCard";
import StateCard from "./components/StateCard";
import SystemCard from "./components/SystemCard";
import type { DaggerStatus, EpisodeEntry } from "./types";

export default function App() {
  const [snap, setSnap] = useState<DaggerStatus | null>(null);
  const [selectedEp, setSelectedEp] = useState<EpisodeEntry | null>(null);
  const [conn, setConn] = useState<"connecting" | "open" | "closed">("connecting");
  const [task, setTask] = useState("Task_A");
  const [tasks, setTasks] = useState<{ task: string; has_data: boolean }[]>([]);
  const [episodes, setEpisodes] = useState<EpisodeEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  // Episode data lives here so EpisodesCard (counts) + HistoryCard (list)
  // share one source — single fetch per task, refreshed when a recording
  // finishes (dagger/inference counts in the WS snapshot bump).
  const reloadEpisodes = useCallback(async () => {
    try { setEpisodes(await api.episodes(task)); } catch { /* backend may be mid-restart */ }
  }, [task]);

  useEffect(() => { api.tasks().then(setTasks).catch(() => {}); }, []);
  useEffect(() => { reloadEpisodes(); }, [reloadEpisodes]);
  useEffect(() => { reloadEpisodes(); /* eslint-disable-next-line */ },
    [snap?.dagger_episodes, snap?.inference_episodes]);

  // WebSocket with auto-reconnect — backend restart shouldn't require browser
  // refresh.
  useEffect(() => {
    let timer: number | null = null;
    let stop = false;
    const open = () => {
      setConn("connecting");
      const ws = connectStatusWs(setSnap);
      wsRef.current = ws;
      ws.onopen = () => setConn("open");
      ws.onclose = () => {
        setConn("closed");
        if (!stop) timer = window.setTimeout(open, 1500);
      };
      ws.onerror = () => { try { ws.close(); } catch {} };
    };
    open();
    return () => {
      stop = true;
      if (timer) clearTimeout(timer);
      try { wsRef.current?.close(); } catch {}
    };
  }, []);

  const state = snap?.state ?? "—";
  const stateCls = state === "—" ? "state-unknown" : `state-${state}`;

  return (
    <div className="app">
      {/* ── top bar: at-a-glance status chips ── */}
      <div className="card top-bar">
        <h1>DAgger Manager</h1>
        <span className={`state-badge ${stateCls}`} style={{ fontSize: 12 }}>{state}</span>
        {snap?.recording && (
          <span className="chip rec"><span className="rec-dot" />REC</span>
        )}
        <span className={`chip ${snap?.session_running ? "on" : ""}`}>
          {snap?.session_running ? "● policy" : "○ no policy"}
        </span>
        <span className={`chip ${snap?.ros_alive ? "on" : ""}`}>
          {snap?.ros_alive ? "● infra" : "○ infra"}
        </span>
        <span className="spacer" />
        <div className="conn">ws: {conn}</div>
      </div>

      {/* ── cameras: full width (within the 1440 cap) so the live preview
             is large — the operator's primary view ── */}
      <CameraGrid cameras={snap?.cameras ?? {}} />

      {/* ── control strip: 3 columns of stacked cards. Column flow keeps
             each column tightly packed (no blank gaps between cards). ── */}
      <div className="ctrl-region">
        <div className="ctrl-col ctrl-col-wide">
          <SystemCard s={snap} />
        </div>
        <div className="ctrl-col">
          <StateCard s={snap} />
          <ControlsCard s={snap} />
        </div>
        <div className="ctrl-col">
          <ArmsPanel />
          <EpisodesCard s={snap} task={task} tasks={tasks}
                        episodes={episodes} onTask={setTask} />
        </div>
      </div>

      {/* ── history + replay region ── */}
      <div className="section-label">History &amp; Replay</div>
      <div className="hr-region">
        <div className="hr-history">
          <HistoryCard task={task} episodes={episodes}
                       selected={selectedEp} onSelect={setSelectedEp}
                       onReload={reloadEpisodes} />
        </div>
        <div className="hr-replay">
          <ReplayCard s={snap} ep={selectedEp} task={task}
                      onDeleted={() => { setSelectedEp(null); reloadEpisodes(); }} />
        </div>
      </div>
    </div>
  );
}
