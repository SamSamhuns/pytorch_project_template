#!/bin/bash

# Default container name
def_cont_name="pytorch_model_server_cont"

# Exit on errors, unset variables, or failed pipes
set -euo pipefail

# Help function
helpFunction() {
    echo ""
    echo "Usage: $0 -p <port>"
    echo -e "\t-p Port number for FastAPI (e.g., 8000)"
    exit 1
}

# Parse arguments
port=""
while getopts "p:" opt; do
    case "$opt" in
        p)
            port="$OPTARG"
            ;;
        ?)
            helpFunction
            ;;
    esac
done

# Validate arguments
if [[ -z "$port" ]]; then
    echo "Error: Port number is required."
    helpFunction
fi

# Ensure port is numeric and within the valid range
if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    echo "Error: Port must be a number between 1 and 65535."
    exit 1
fi

# Trap to handle script interruption (e.g., Ctrl+C)
cleanup() {
    echo "Script interrupted. Cleaning up..."
    if [ "$(docker ps -q -f name=$def_cont_name)" ]; then
        echo "Stopping and removing container '$def_cont_name'..."
        docker stop "$def_cont_name" >/dev/null 2>&1 || true
        docker rm "$def_cont_name" >/dev/null 2>&1 || true
    fi
    exit
}
trap cleanup INT TERM

# Check if the container is running and stop it if necessary
if [ "$(docker ps -q -f name=$def_cont_name)" ]; then
    echo "Stopping and removing existing container '$def_cont_name'..."
    docker stop "$def_cont_name"
    docker rm "$def_cont_name"
fi

# Ensure the checkpoints directory exists
checkpoints_dir="$PWD/checkpoints_server"
if [ ! -d "$checkpoints_dir" ]; then
    echo "Creating checkpoints directory at $checkpoints_dir..."
    mkdir -p "$checkpoints_dir"
fi

# Start the Docker container
echo "Starting Docker container '$def_cont_name' with FastAPI on port: $port..."
docker run \
    -ti --rm \
    -p 0.0.0.0:"$port":8080 \
    -v "$checkpoints_dir:/home/user1/app/checkpoints_server" \
    --name "$def_cont_name" \
    --env LANG=en_US.UTF-8 \
    pytorch_model_server \
    bash
