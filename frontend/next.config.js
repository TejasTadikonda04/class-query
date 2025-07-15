/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Opt‑in to the (optional) “appDir” experiment
  // Disable it here so we can use the classic pages/ directory.
  experimental: {
    appDir: false
  }
};

module.exports = nextConfig;
