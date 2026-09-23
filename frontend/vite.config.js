import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend port was hardcoded here, so start.ps1 -ApiPort moved uvicorn but
// left the proxy pointing at 8000: the UI loaded and every /api call failed.
// Reading it from the environment keeps the default identical while letting the
// launcher move both halves together.
const apiPort = process.env.VITE_API_PORT || "8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": `http://127.0.0.1:${apiPort}`
    }
  },
  // The honesty checklist (docs/ui_simplification_plan.md D3) lives under
  // test/, never under src/: tests/test_founder_contract.py scans every file
  // there for engine vocabulary, and a cycle fixture would trip it.
  test: {
    environment: "jsdom",
    include: ["test/**/*.test.jsx"]
  }
});
