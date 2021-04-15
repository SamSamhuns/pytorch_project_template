#!/bin/bash
docker build -t pytorch_model --build-arg ENV=$1 .

# bash build_docker.sh train
# bash build_docker.sh test
