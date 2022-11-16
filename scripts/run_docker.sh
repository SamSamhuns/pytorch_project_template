#!/bin/bash

docker run \
      -ti --rm \
      -p 0.0.0.0:6006:6006 \
      -v "$PWD"/checkpoints_docker:/pytorch_model/checkpoints \
      -v "$PWD"/data:/pytorch_model/data \
      --name pytorch_container \
      pytorch_model \
      bash
