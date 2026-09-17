/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output -- a real, disclosed choice: keeps the Docker image to the built
  // server + its own pruned node_modules subset, not the full dev toolchain. See
  // retail_frontend/Dockerfile's own comment.
  output: "standalone",
  reactStrictMode: true,
};

module.exports = nextConfig;
