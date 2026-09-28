/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Pure static viewer over the exported JSON — no backend.
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  // This app lives inside a larger Python repository, so pin the Turbopack root
  // here rather than letting Next.js walk up and infer the monorepo root.
  turbopack: { root: import.meta.dirname },
};

export default nextConfig;
