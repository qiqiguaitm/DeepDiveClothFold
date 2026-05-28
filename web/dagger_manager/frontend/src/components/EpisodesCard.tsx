import type { DaggerStatus, EpisodeEntry } from "../types";

interface Props {
  s: DaggerStatus | null;
  task: string;
  tasks: { task: string; has_data: boolean }[];
  episodes: EpisodeEntry[];
  onTask: (t: string) => void;
}

export default function EpisodesCard({ s, task, tasks, episodes, onTask }: Props) {
  const inf = episodes.filter(e => e.subset === "inference").length;
  const dag = episodes.filter(e => e.subset === "dagger").length;
  const frames = episodes.reduce((a, e) => a + e.length, 0);
  const dur = episodes.reduce((a, e) => a + e.duration_s, 0);

  return (
    <div className="card eps-card">
      <h2>Episodes</h2>
      <div className="kv">
        <div className="k">Task</div>
        <div className="v">
          <select className="select" value={task} onChange={e => onTask(e.target.value)}>
            {tasks.length === 0 && <option value={task}>{task}</option>}
            {tasks.map(t => (
              <option key={t.task} value={t.task}>
                {t.task}{t.has_data ? "" : " · empty"}
              </option>
            ))}
          </select>
        </div>
        <div className="k">DAgger</div>
        <div className="v"><b style={{ color: "#f0c674" }}>{dag}</b> corrections</div>
        <div className="k">Inference</div>
        <div className="v"><b style={{ color: "#79c0ff" }}>{inf}</b> rollouts</div>
        <div className="k">Total</div>
        <div className="v">{frames.toLocaleString()} frames · {(dur / 60).toFixed(1)} min</div>
      </div>
      <div className="hint">
        Form C dual-dataset — both subsets feed RECAP advantage training.
        {s?.session_running && s?.task && s.task !== task && (
          <> · live session is on <b>{s.task}</b></>
        )}
      </div>
    </div>
  );
}
