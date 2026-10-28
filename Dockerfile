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
ENV UV_PYTHON_DOWNLOADS=never

# install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

# set workdir and create with proper ownership
ENV WORKDIR="/home/${UNAME}/pytorch_model"
RUN mkdir -p ${WORKDIR} && chown -R ${UID}:${GID} /home/${UNAME}
WORKDIR ${WORKDIR}

# copy the dependency files first for better layer caching
COPY --chown=${UID}:${GID} pyproject.toml uv.lock ${WORKDIR}/

# switch to user before installing dependencies
USER ${UNAME}

# install dependencies with CUDA torch, plus the dev and test groups
RUN uv sync --locked --extra cu130 --no-cache

# copy the rest of the project files
COPY --chown=${UID}:${GID} . ${WORKDIR}/

# activate virtual environment by default
ENV VIRTUAL_ENV="${WORKDIR}/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# set default command
CMD ["/bin/bash"]