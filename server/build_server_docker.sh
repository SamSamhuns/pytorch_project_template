#!/bin/bash

# check if 2 cmd args have been entered
if [ "$#" -ne 2 ]
  then
    echo "Mode must be specified for creating Dockerfile"
		echo "eg. \$ bash build_docker.sh -m pytorch/onnx"
		exit
fi

# check if -m/--mode flag has been entered
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--mode) mode="$2"; shift ;;
        *) echo "Unknown parameter: $1";
	exit 1 ;;
    esac
    shift
done

echo "Building Docker Container with $mode inference mode"
docker build -t pytorch_model_server -f server/Dockerfile --build-arg MODE="$mode" --build-arg UID=$(id -u) .
