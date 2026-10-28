# Use PyTorch base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Define build arguments with defaults
ARG UNAME=user1
ARG UID=1000
ARG CUDA="12.4"
ARG CUDNN="9.0"

# Metadata
LABEL maintainer="fname.lname@domain.com"

# Set environment variables
ENV WORKDIR="/home/$UNAME/app"
ENV VIRTUAL_ENV="/home/$UNAME/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install basic dependencies
RUN apt-get update -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
    apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ vim \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -rm --home-dir "/home/$UNAME" --shell /bin/bash -u "$UID" "$UNAME"

# Set working directory
WORKDIR ${WORKDIR}

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

# Install the dependencies (CPU torch and the server group) into $VIRTUAL_ENV
COPY pyproject.toml uv.lock "$WORKDIR/"
RUN UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV" uv sync --locked --extra cpu --group server --no-dev --no-cache

# Copy application code to the working directory
COPY . "$WORKDIR"

# Change file ownership directly while copying
RUN chown -R "$UNAME" "$WORKDIR"

# Switch to non-root user
USER "$UNAME"

# Default command
CMD ["python3", "server/server.py"]
