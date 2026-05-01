import { rmSync } from "node:fs";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

function pruneLargePublicAssets() {
  return {
    name: "prune-large-public-assets",
    closeBundle() {
      const largeOst = new URL("./dist/palace_ost.mp3", import.meta.url);
      try {
        rmSync(largeOst, { force: true });
      } catch (_error) {
        // Build pruning is best-effort; Vercel ignore rules still exclude it.
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), pruneLargePublicAssets()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
