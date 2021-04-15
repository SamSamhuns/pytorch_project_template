FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ARG CUDA="10.1"
ARG CUDNN="7.6"

# Maintainer
LABEL maintainer="fname.lname@domain.com"

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ vim \
 && apt-get install -y libsm6 libxext6 libgl1-mesa-glx

# set work directory
WORKDIR /app

# create req sub-dirs inside app
RUN  mkdir -p /app/modules \
  && mkdir -p /app/configs \
  && mkdir -p /app/checkpoints

# Install python dependencies
RUN pip install --upgrade pip
ARG req_path="../requirements/server.txt"
COPY $req_path /app/
RUN pip install -r server.txt

# copy req files to app
COPY ../checkpoints /app/checkpoints
COPY ../modules /app/modules
COPY ../configs /app/configs
COPY ../inference.py /app
COPY server.py /app
CMD ["python3", "server.py"]
