import type { CameraHealth } from "../types";

interface Props { cameras: Record<string, CameraHealth>; }

// Tile order + labels match start_data_collect.sh's data_manager UI.
const TILES: { key: string; label: string }[] = [
  { key: "hand_left", label: "左腕 hand_left (D405)" },
  { key: "top_head", label: "头部 top_head (D435)" },
  { key: "hand_right", label: "右腕 hand_right (D405)" },
];

export default function CameraGrid({ cameras }: Props) {
  return (
    <div className="card cams-card">
      <h2>相机预览</h2>
      <div className="cam-grid">
        {TILES.map((t) => {
          const h = cameras?.[t.key];
          const live = h && h.fps > 0;
          return (
            <div key={t.key} className="cam-tile">
              <span className="cam-label">{t.label}</span>
              {live ? (
                <img
                  src={`/api/camera/${t.key}/mjpeg`}
                  alt={t.key}
                  style={{ width: "100%", background: "#000", borderRadius: 4, display: "block" }}
                />
              ) : (
                <span style={{ color: "#8b949e" }}>● 等待 ROS2 视频流…</span>
              )}
              {h && (
                <span className="cam-stat">
                  {h.fps} fps · {h.latency_ms} ms · drop {h.dropped}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
