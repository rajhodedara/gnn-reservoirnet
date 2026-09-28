import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#06090b",
          900: "#080d11",
          850: "#0a1116",
          800: "#0d151b",
          700: "#131e26",
          600: "#1b2a33",
        },
        hair: "rgba(150,180,190,0.14)",
        hairStrong: "rgba(150,180,190,0.26)",
        data: {
          text: "#c9d6da",
          dim: "#7b9099",
          faint: "#4d6169",
          teal: "#35c2b0",
          tealDim: "#1c6f66",
          amber: "#e9a13b",
          amberDim: "#8a5f1c",
          rose: "#c9615f",
          roseDim: "#7a3735",
          indigo: "#7d8fe6",
          sky: "#5aa9d6",
        },
      },
      fontFamily: {
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "SF Mono",
          "Menlo",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
      },
      fontSize: {
        // A real scale. The old config only had 2xs, which is why every
        // element collapsed into the same tiny size.
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
        xs: ["0.75rem", { lineHeight: "1.05rem" }],
        sm: ["0.8125rem", { lineHeight: "1.15rem" }],
        base: ["0.875rem", { lineHeight: "1.35rem" }],
        lg: ["1rem", { lineHeight: "1.4rem" }],
        xl: ["1.125rem", { lineHeight: "1.45rem" }],
        "2xl": ["1.375rem", { lineHeight: "1.5rem" }],
        "3xl": ["1.75rem", { lineHeight: "1.6rem" }],
      },
      letterSpacing: {
        label: "0.13em",
      },
    },
  },
  plugins: [],
};

export default config;
