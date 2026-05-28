import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Port 5174 (data_manager already runs on 5173). Backend at 8788.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    strictPort: true,
    proxy: {
      "/api": "http://localhost:8788",
      "/ws": { target: "ws://localhost:8788", ws: true },
    },
  },
});
