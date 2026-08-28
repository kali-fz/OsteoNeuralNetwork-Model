import { defineConfig } from "vite";

/**
 * The ONNM frontend build.
 *
 * Output goes to web/dist, which wrangler.jsonc serves as the Worker's static
 * assets. `/api/*` never reaches the asset layer -- `run_worker_first` in that
 * file routes it to the Worker first -- so there is no dev/prod difference in
 * how the API is addressed.
 */
export default defineConfig({
  root: "web",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    // The globe's world atlas is 107 KB of JSON. Inlining it as a data URI
    // would bloat the entry chunk and delay first paint, so it stays a
    // separately cached asset.
    assetsInlineLimit: 4096,
  },
  server: {
    // `npm run dev` serves the frontend; the API is proxied to the deployed
    // Worker so a local UI session talks to the real D1 rather than a stub.
    proxy: {
      "/api": {
        target: "https://onnm.kali-fz.workers.dev",
        changeOrigin: true,
        secure: true,
      },
    },
  },
});
