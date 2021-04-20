FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG CUDA="11.1"
ARG CUDNN="8.0"

# maintainer
LABEL maintainer="fname.lname@domain.com"

# install basics
RUN apt-get  update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ vim\
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# set work directory
WORKDIR /pytorch_model

# install python dependencies
COPY ./requirements/train.txt /pytorch_model/
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 -r train.txt

# freq changing files are added below
COPY . /pytorch_model
CMD ["python3", "train_mnist.py"]
