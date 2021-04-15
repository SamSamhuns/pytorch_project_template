#!/bin/bash

docker run \
      -ti \
      -p 0.0.0.0:8000:8080 \
      --name pytorch_project \
      --env LANG=en_US.UTF-8 \
      --gpus '"device=0"' \
      img_name:tag
