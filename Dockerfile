# PyPI torch wheels bundle the CUDA runtime and cuDNN, so a slim Python base is enough;
# the NVIDIA driver comes from the host through the NVIDIA Container Toolkit
FROM python:3.13-slim

# install uv
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

# set username & uid inside docker
ARG UNAME=user1
ARG UID=1000
ARG GID=1000

# create group and user with proper permissions
RUN groupadd -g "${GID}" "${UNAME}" \
    && useradd -m --shell /bin/bash -u "${UID}" -g "${GID}" "${UNAME}"

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_PYTHON_DOWNLOADS=never
# the uv cache is a mount, so copy files out of it instead of hard-linking
ENV UV_LINK_MODE=copy

# torch build to install: cu130 for NVIDIA GPUs (driver >= 580) or cpu
ARG TORCH_EXTRA=cu130

# set workdir and create with proper ownership
ENV WORKDIR="/home/${UNAME}/pytorch_model"
WORKDIR ${WORKDIR}
RUN chown ${UID}:${GID} ${WORKDIR}
USER ${UNAME}

# install the dependencies first for better layer caching
COPY --chown=${UID}:${GID} pyproject.toml uv.lock ${WORKDIR}/
RUN --mount=type=cache,target=/home/${UNAME}/.cache/uv,uid=${UID},gid=${GID} \
    uv sync --locked --extra ${TORCH_EXTRA} --no-dev

# copy the rest of the project files
COPY --chown=${UID}:${GID} . ${WORKDIR}/

# activate virtual environment by default
ENV VIRTUAL_ENV="${WORKDIR}/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"

# set default command
CMD ["/bin/bash"]
