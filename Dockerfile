FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG CUDA="12.4"
ARG CUDNN="9.0"
ARG DEBIAN_FRONTEND=noninteractive

# maintainer
LABEL maintainer="fname.lname@domain.com"

# install basics and clean in single layer to reduce image size
RUN apt-get update -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    git \
    curl \
    ca-certificates \
    bzip2 \
    cmake \
    tree \
    htop \
    bmon \
    iotop \
    g++ \
    vim \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# set username & uid inside docker
ARG UNAME=user1
ARG UID=1000
ARG GID=1000

# create group and user with proper permissions
RUN groupadd -g "${GID}" "${UNAME}" \
    && useradd -rm --home-dir "/home/${UNAME}" --shell /bin/bash -g "${UNAME}" -G sudo -u "${UID}" "${UNAME}"

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV POETRY_VERSION=2.1.3
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1
ENV POETRY_CACHE_DIR="/tmp/poetry_cache"
ENV PATH="${POETRY_HOME}/bin:$PATH"

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && chmod a+x ${POETRY_HOME}/bin/poetry

# set workdir and create with proper ownership
ENV WORKDIR="/home/${UNAME}/pytorch_model"
RUN mkdir -p ${WORKDIR} && chown -R ${UID}:${GID} /home/${UNAME}
WORKDIR ${WORKDIR}

# copy poetry files first for better layer caching
COPY --chown=${UID}:${GID} pyproject.toml poetry.lock* ${WORKDIR}/

# switch to user before installing dependencies
USER ${UNAME}

# install dependencies using poetry (without installing the project itself)
RUN poetry install --no-root --no-ansi --with dev,test \
    && poetry cache clear pypi --all

# copy the rest of the project files
COPY --chown=${UID}:${GID} . ${WORKDIR}/

# activate virtual environment by default
ENV VIRTUAL_ENV="${WORKDIR}/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# set default command
CMD ["/bin/bash"]