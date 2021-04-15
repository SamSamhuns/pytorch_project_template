FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# ENV should be train or test
ARG ENV
ARG CUDA="10.1"
ARG CUDNN="7.6"

# maintainer
LABEL maintainer="fname.lname@domain.com"

# install basics
RUN apt-get  update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ vim\
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# set work directory
WORKDIR /pytorch_model

# install python dependencies
RUN pip install --upgrade pip
ARG req_path="./requirements/${ENV}.txt"
COPY $req_path /pytorch_model/
RUN pip install -r $ENV.txt

# freq changing files are added below
COPY . /pytorch_model
CMD ["python3", "train_mnist.py"]
