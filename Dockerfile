FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG CUDA="11.7"
ARG CUDNN="8.0"

# maintainer
LABEL maintainer="fname.lname@domain.com"

# install basics
RUN apt-get  update -y --no-install-recommends \
 && apt-get install -y --no-install-recommends apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ vim \
 && apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && apt-get clean

# remove cache
RUN apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# set username & uid inside docker
ARG UNAME=user1
ARG UID=1000
ENV WORKDIR="/home/$UNAME/pytorch_model"

# add user UNAME as a member of the sudoers group
RUN useradd -rm --home-dir "/home/$UNAME" --shell /bin/bash -g root -G sudo -u "$UID" "$UNAME"

# set workdir
WORKDIR ${WORKDIR}

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# setup virtual env for python
ENV VIRTUAL_ENV="/home/$UNAME/venv"
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python dependencies
RUN pip install pip==23.1.2
COPY ./requirements/train.txt "$WORKDIR/train.txt"
RUN pip install --no-cache-dir --default-timeout=100 -r "$WORKDIR/train.txt"

# freq changing files are added below
COPY . "$WORKDIR"

# change file ownership to docker user
RUN chown -R "$UNAME" "$WORKDIR"

USER "$UNAME"
