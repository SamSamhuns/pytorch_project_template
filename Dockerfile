# need pytorch 1.7.0
# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
FROM ubuntu:18.04

# ENV should be train or test
ARG ENV
ARG CUDA="10.0"
ARG CUDNN="7.6"

# Maintainer
LABEL maintainer="fname.lname@domain.com"

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ vim\
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# set work directory
WORKDIR /app

# Install python dependencies
RUN pip install --upgrade pip
ARG req_path="./requirements/requirements-${ENV}.txt"
COPY $req_path /app/
RUN pip install -r requirements-$ENV.txt

COPY . /app
CMD ["python3", "server.py"]
