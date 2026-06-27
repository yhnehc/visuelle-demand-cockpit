import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/lmstudio": {
        target: "http://127.0.0.1:1234",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/lmstudio/, "/v1"),
      },
    },
  },
});
