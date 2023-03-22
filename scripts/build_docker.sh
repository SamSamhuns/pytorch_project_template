#!/bin/bash
docker build -t pytorch_model:latest --build-arg UID=$(id -u) .
