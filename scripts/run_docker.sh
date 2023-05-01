#!/bin/bash

def_cont_name=pytorch_model_train_cont
port=6006

helpFunction()
{
   echo ""
   echo "Usage: $0 -p port"
   echo -e "\t-p http_port"
   exit 1 # Exit script after printing help
}

while getopts "p:" opt
do
   case "$opt" in
      p ) port="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$port" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Check if the container is running
if [ "$(docker ps -q -f name=$def_cont_name)" ]; then
    echo "Stopping docker container '$def_cont_name'"
    docker stop "$def_cont_name"
    docker rm "$def_cont_name"
    echo "Stopped & removed container '$def_cont_name'"
fi

# 6006 is tensorboard default port
mkdir -p checkpoints_docker  # create checkpoints_docker dir so that it has proper perm
docker run \
      -ti --rm \
      -p "0.0.0.0:$port:6006" \
      --gpus "device=0" \
      -v "$PWD"/checkpoints_docker:/home/user1/pytorch_model/checkpoints \
      -v "$PWD"/data:/home/user1/pytorch_model/data \
      --name "$def_cont_name" \
      pytorch_model:latest \
      bash
