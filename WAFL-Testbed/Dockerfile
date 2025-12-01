# Optimized Dockerfile for faster builds with BuildKit cache

# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

# ==========================================
# Stage 1: System dependencies
# ==========================================
FROM base AS system-deps

# Install system dependencies in a single layer
# Use --mount=type=cache for apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
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
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy only dependency files first (for better layer caching)
COPY pyproject.toml uv.lock ./

# Copy wafl directory structure (needed for package installation)
COPY wafl/__init__.py ./wafl/__init__.py
COPY wafl/src/__init__.py ./wafl/src/__init__.py

# Install dependencies with cache mount for uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ==========================================
# Stage 3: Final image
# ==========================================
FROM python-deps AS final

# Copy the rest of the application
# This layer changes frequently, so keep it last
COPY wafl/ ./wafl/
COPY data/ ./data/
COPY ctrl/ ./ctrl/

# Expose ports
# 10001: Control Port
# 10002: P2P Port
EXPOSE 10001 10002

# Default command (can be overridden)
CMD ["uv", "run", "python", "-m", "src.common.main"]
