# Slim Python base; the server runs on CPU torch
FROM python:3.13-slim

# Define build arguments with defaults
ARG UNAME=user1
ARG UID=1000

# Set environment variables
ENV WORKDIR="/home/$UNAME/app"
ENV VIRTUAL_ENV="/home/$UNAME/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PYTHON_DOWNLOADS=never

# Create non-root user
RUN useradd -m --shell /bin/bash -u "$UID" "$UNAME"

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
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]
