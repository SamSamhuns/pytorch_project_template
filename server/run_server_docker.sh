#!/bin/bash

def_cont_name=pytorch_model_server_cont

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

echo "Docker Container starting with FastAPI port: $port"
mkdir -p checkpoints_server  # create checkpoints_server dir so that it has proper perm
docker run \
      -ti --rm \
      -p 0.0.0.0:"$port":8080 \
      -v "$PWD"/checkpoints_server:/home/user1/app/checkpoints_server \
      --name "$def_cont_name" \
      --env LANG=en_US.UTF-8 \
      pytorch_model_server \
      bash
