#!/bin/bash

# check if 2 cmd args have been entered
if [ "$#" -ne 2 ]
  then
    echo "Server inference mode (pytorch/onnx) must be specified"
		echo "eg. \$ bash run_server.sh -m pytorch/onnx"
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

echo "Starting Model Server in Docker Container in $mode mode"

# check if the correct mode has been selected
if [ "$mode" == "pytorch" ]
then
   # copy files for pytorch inference with fastapi
   cp -r src server/
   cp -r configs server/
   cp -r checkpoints server/
elif [ "$mode" == "onnx" ]
then
   # copy files for onnx inference with fastapi
   cp -r configs server/
   cp -r checkpoints server/
else
   echo "$mode is not recognized. Only pytorch/onnx modes are allowed"
fi

# cd to server dir and create docker file and run dockerfile to start server
cd server
bash build_docker.sh -m "$mode"
bash run_docker.sh -h 8008

# server will be avai at localhost:8008
