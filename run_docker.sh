#!/bin/bash

docker run \
      -ti \
      -p 0.0.0.0:6006:6006 \
      -v checkpoints:/pytorch_model/checkpoints \
      --name pytorch_container \
      --env LANG=en_US.UTF-8 \
      pytorch_model \
      bash
