#!/bin/bash

# check if 2 cmd args have been entered
if [ $# -ne 2 ]
  then
    echo "HTTP port must be specified for FastAPI"
		echo "eg. \$ bash run_docker.sh -h 8080"
		exit
fi

# check if -h/--http flag has been entered
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--http) http="$2"; shift ;;
        *) echo "Unknown parameter: $1";
	exit 1 ;;
    esac
    shift
done

echo "Docker Container starting with FastAPI port: $http"

docker run \
      -ti --rm \
      -p 0.0.0.0:$http:8080 \
      --name model_server \
      --env LANG=en_US.UTF-8 \
      model_server \
      bash
