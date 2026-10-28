#!/bin/bash
# extra docker build args are passed through, e.g. --build-arg TORCH_EXTRA=cpu for a CPU-only image
docker build -t pytorch_model:latest --build-arg UID=$(id -u) --build-arg GID=$(id -g) "$@" .
