/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: "200mb",
    },
  },
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
