import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0c0f",
        surface: "#14161b",
        border: "#262932",
        muted: "#8a90a0",
        text: "#e6e8ee",
        accent: "#6ee7b7",
        warn: "#fbbf24",
        bad: "#ef4444",
      },
    },
  },
  plugins: [],
};

export default config;
