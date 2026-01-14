# syntax=docker/dockerfile:1
# Optimized Dockerfile for faster builds with BuildKit cache
FROM python:3.11-slim@sha256:d963377c651e0ba71336362d37cb32531e9a569407c7bea07e2a05a88dde145d AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_HTTP_TIMEOUT=3600 \
    UV_REQUEST_TIMEOUT=3600 \
    HTTP_TIMEOUT=3600

# ==========================================
# Stage 1: System dependencies
# ==========================================
FROM base AS system-deps

# Install system dependencies in a single layer
# Use --mount=type=cache for apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    iproute2 \
    iputils-ping \
    dstat \
    sysstat \
    procps \
    iperf3 \
    stress-ng

# ==========================================
# Stage 2: Python dependencies
# ==========================================
FROM system-deps AS python-deps

# Install uv (copy from official image)
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:816fdce3387ed2142e37d2e56e1b1b97ccc1ea87731ba199dc8a25c04e4997c5 /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy only dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Copy wafl directory structure (needed for package installation)
COPY wafl/__init__.py ./wafl/__init__.py
COPY wafl/src/__init__.py ./wafl/src/__init__.py

# Install dependencies with cache mount for uv
RUN --mount=type=cache,id=wafl-uv-cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --compile-bytecode

# ==========================================
# Stage 3: Final image
# ==========================================
FROM python-deps AS final

# Copy the rest of the application
# This layer changes frequently, so keep it last
# NOTE: data/ is NOT copied - it's transferred via rsync and mounted at runtime
COPY wafl/ ./wafl/
COPY utils/ ./utils/

# Expose ports
# 10001: Control Port
# 10002: P2P Port
EXPOSE 10001 10002

# Default command (can be overridden)
CMD ["uv", "run", "python", "-m", "src.common.main"]
