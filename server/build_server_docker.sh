#!/bin/bash

# Exit on error and treat unset variables as an error
set -euo pipefail

# Function to display usage
usage() {
    echo "Usage: $0"
    exit 1
}

# Ensure no arguments are provided
if [ "$#" -ne 0 ]; then
    echo "Error: No arguments are expected."
    usage
fi

# Build the Docker container
echo "Building Docker container for inference ..."
docker build -t pytorch_model_server -f server/Dockerfile \
    --build-arg UID="$(id -u)" .
