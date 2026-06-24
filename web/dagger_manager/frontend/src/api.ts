import type { CkptEntry, DaggerStatus, EpisodeEntry, JointState } from "./types";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch {}
    throw new Error(`${res.status} ${res.statusText}${detail ? ": " + detail : ""}`);
  }
  return res.json();
}

export const api = {
  status: () => json<DaggerStatus>("/api/dagger/status"),
  ckpts: () => json<CkptEntry[]>("/api/dagger/ckpts"),
  stackStart: (body: { ckpt?: string; task?: string; subset?: string; prompt?: string }) =>
    json("/api/dagger/stack/start", { method: "POST", body: JSON.stringify(body) }),
  stackStop: () => json("/api/dagger/stack/stop", { method: "POST" }),
  sessionStart: (body: { ckpt: string; gpu_id?: string; prompt?: string; variant?: string }) =>
    json("/api/dagger/session/start", { method: "POST", body: JSON.stringify(body) }),
  sessionStop: () => json("/api/dagger/session/stop", { method: "POST" }),
  systemStart: (body: { ckpt: string; gpu_id?: string; prompt?: string; variant?: string }) =>
    json("/api/dagger/system/start", { method: "POST", body: JSON.stringify(body) }),
  systemStop: () => json("/api/dagger/system/stop", { method: "POST" }),
  takeover: (enable: boolean) =>
    json("/api/dagger/takeover", { method: "POST", body: JSON.stringify({ enable }) }),
  recordToggle: () => json("/api/dagger/record/toggle", { method: "POST" }),
  recordStart: () => json("/api/dagger/record/start", { method: "POST" }),
  recordSave: () => json("/api/dagger/record/save", { method: "POST" }),
  recordDiscard: () => json("/api/dagger/record/discard", { method: "POST" }),
  execute: (enable: boolean) =>
    json("/api/dagger/execute", { method: "POST", body: JSON.stringify({ enable }) }),
  rolloutNext: () => json("/api/dagger/rollout/next", { method: "POST" }),
  joints: () => json<JointState>("/api/joints"),
  tasks: () => json<{ task: string; has_data: boolean }[]>("/api/dagger/tasks"),
  episodes: (task = "Task_A") =>
    json<EpisodeEntry[]>(`/api/dagger/episodes?task=${encodeURIComponent(task)}`),
  delEpisode: (subset: string, date: string, ep: number, task = "Task_A") =>
    json(`/api/dagger/episodes/${subset}/${date}/${ep}?task=${encodeURIComponent(task)}`,
         { method: "DELETE" }),
  // Video URL (not fetched as JSON — used as <video src>). Vite proxies /api.
  // `bust` (episode created_at) cache-busts: re-recording an episode at the same
  // subset/date/id reuses the URL, so without this the browser replays the stale
  // cached video (the endpoint sends no Cache-Control).
  episodeVideoUrl: (subset: string, date: string, ep: number, camera: string,
                    task = "Task_A", bust?: number | null) =>
    `/api/dagger/episodes/${subset}/${date}/${ep}/video/${camera}?task=${encodeURIComponent(task)}`
    + (bust != null ? `&t=${bust}` : ""),
};

export function connectStatusWs(onSnap: (s: DaggerStatus) => void): WebSocket {
  // Vite dev server proxies /ws to ws://localhost:8788
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/dagger`);
  ws.onmessage = (ev) => {
    try {
      onSnap(JSON.parse(ev.data) as DaggerStatus);
    } catch {}
  };
  return ws;
}
