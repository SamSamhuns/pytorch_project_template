#!/bin/bash

# Exit on error and treat unset variables as an error
set -euo pipefail

# Function to display usage
usage() {
    echo "Usage: $0 -m <mode>"
    echo "Example: $0 -m pytorch/onnx"
    exit 1
}

# Ensure exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments provided."
    usage
fi

# Parse arguments
mode=""
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            if [[ -n "${2:-}" ]]; then
                mode="$2"
                shift
            else
                echo "Error: Missing value for the -m/--mode argument."
                usage
            fi
            ;;
        *)
            echo "Error: Unknown parameter: $1"
            usage
            ;;
    esac
    shift
done

# Ensure mode is not empty
if [[ -z "$mode" ]]; then
    echo "Error: Mode argument cannot be empty."
    usage
fi

# Build the Docker container
echo "Building Docker container with $mode inference mode..."
docker build -t pytorch_model_server -f server/Dockerfile \
    --build-arg MODE="$mode" \
    --build-arg UID="$(id -u)" .
