# Use PyTorch base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Define build arguments with defaults
ARG MODE=dev
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

# Setup Python virtual environment
RUN python3 -m venv $VIRTUAL_ENV

# Install and pin pip to the latest version
RUN pip install pip==24.0

# Copy and install Python dependencies
COPY ./server/requirements_${MODE}_server.txt "$WORKDIR/"
RUN pip install -r "$WORKDIR/requirements_${MODE}_server.txt"

# Copy application code to the working directory
COPY . "$WORKDIR"

# Change file ownership directly while copying
RUN chown -R "$UNAME" "$WORKDIR"

# Switch to non-root user
USER "$UNAME"

# Default command
CMD ["python3", "server/server.py"]
