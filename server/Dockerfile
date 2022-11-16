FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ARG MODE
ARG CUDA="10.1"
ARG CUDNN="7.6"

# Maintainer
LABEL maintainer="fname.lname@domain.com"

# install basics
RUN apt-get update -y --no-install-recommends \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ vim \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# set work directory
WORKDIR /app

# setup virtual env for python
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install python dependencies
RUN pip install --upgrade pip
COPY requirements_${MODE}_server.txt /app/
RUN pip install -r requirements_${MODE}_server.txt

# copy req files to app
COPY . /app/
CMD ["python3", "server.py"]
