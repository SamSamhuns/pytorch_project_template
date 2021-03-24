#!/bin/bash
docker build -t img_name --build-arg ENV=$1 .

# bash build_docker.sh train
# bash build_docker.sh test
