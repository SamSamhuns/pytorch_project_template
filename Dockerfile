FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

# resolve GPG error
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

ARG CUDA="11.1"
ARG CUDNN="8.0"

# maintainer
LABEL maintainer="fname.lname@domain.com"

# install basics
RUN apt-get  update -y --no-install-recommends \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ vim\
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# set work directory
WORKDIR /pytorch_model

# setup virtual env for python
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies
COPY ./requirements/train.txt /pytorch_model/
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 -r train.txt

# freq changing files are added below
COPY . /pytorch_model
