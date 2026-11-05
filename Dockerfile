# syntax=docker/dockerfile:1

ARG CUDA_VERSION=13.0.1

# Build stage: native modules and the CUDA execution-provider binaries.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg python3 make g++ libgomp1 \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install --global pnpm@10.33.0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
# The postinstall script installs the CUDA 12 execution-provider binaries.
ENV ONNXRUNTIME_NODE_INSTALL=cuda12
RUN pnpm install --frozen-lockfile

# Runtime stage: CUDA runtime + cuDNN + Node.
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg libgomp1 \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV NODE_ENV=production
COPY --from=deps /app/node_modules ./node_modules
COPY package.json pnpm-lock.yaml tsconfig.json ./
COPY labeler ./labeler
RUN mkdir -p /app/data && chown -R ubuntu:ubuntu /app
USER ubuntu
EXPOSE 4199 4113
CMD ["./node_modules/.bin/tsx", "labeler/src/index.ts"]
