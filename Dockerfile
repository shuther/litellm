# syntax=docker/dockerfile:1
# Keep this syntax directive! It's used to enable Docker BuildKit
FROM python:3.10-slim as python-base
# FROM python:3.9.17-slim as python-base
#slim for now allows us to save 500Mb of data

#Source: https://gist.github.com/nanmu42/57db1e016bb5c8e326d096c44f8aa93e
#i.e.
#prod: docker build -t litellm-b . --progress=plain
#dev: docker build -t litellm-b . --progress=plain --target development
#dev: docker build -t litellm-b . --progress=plain --target production
ARG PORT=8000
#ENV NODE_ENV=production
#RUN if [ "x$mode" = "xdev" ] ; then echo "Development" ; else echo "Production" ; fi

    # Python
ENV PYTHONUNBUFFERED=1 \
    # pip
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # Poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.6.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    # never create virtual environment automaticly, only use env prepared by us
    POETRY_VIRTUALENVS_CREATE=false \
    \
    # this is where our requirements + virtual environment will live
    VIRTUAL_ENV="/venv"
#    \
#    # Node.js major version. Remove if you don't need.
#    NODE_MAJOR=18

# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VIRTUAL_ENV/bin:$PATH"

# prepare virtual env
RUN python -m venv $VIRTUAL_ENV

# working directory and Python path
WORKDIR /app
ENV PYTHONPATH="/app:$PYTHONPATH"

# pretrained models cache path. Remove if you don't need.
# ref: https://huggingface.co/docs/transformers/installation?highlight=transformers_cache#caching-models
#ENV TRANSFORMERS_CACHE="/opt/transformers_cache/"

################################
# BUILDER-BASE
# Used to build deps + create our virtual environment
################################
FROM python-base as builder-base
RUN apt-get update && \
    apt-get install -y \
    apt-transport-https \
    gnupg \
    ca-certificates \
    build-essential \
    git \
    nano \
    curl && \
    rm -rf /var/lib/apt/lists/*
# rm could not be needed

# install Node.js. Remove if you don't need.
#RUN mkdir -p /etc/apt/keyrings && \
#    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
#    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list && \
#    apt-get update && apt-get install -y nodejs

# install poetry - respects $POETRY_VERSION & $POETRY_HOME
# The --mount will mount the buildx cache directory to where
# Poetry and Pip store their cache so that they can re-use it
RUN --mount=type=cache,target=/root/.cache \
    curl -sSL https://install.python-poetry.org | python -

# used to init dependencies
WORKDIR /app
COPY poetry.lock pyproject.toml ./

COPY . /app
#COPY scripts scripts/
#COPY my_awesome_ai_project/ my_awesome_ai_project/

# install runtime deps to VIRTUAL_ENV
#RUN --mount=type=cache,target=/root/.cache \
#    poetry install --no-root --only main

RUN --mount=type=cache,target=/root/.cache \
    poetry install --no-root --with utils,server --without dev --compile

# populate Huggingface model cache. Remove if you don't need.
#RUN poetry run python scripts/bootstrap.py

# build C dependencies. Remove if you don't need.
#RUN --mount=type=cache,target=/app/scripts/vendor \
#    poetry run python scripts/build-c-denpendencies.py && \
#    cp scripts/lib/*.so /usr/lib

################################
# DEVELOPMENT
# Image used during development / testing
################################
FROM builder-base as development

WORKDIR /app

# quicker install as runtime deps are already installed
RUN --mount=type=cache,target=/root/.cache \
    poetry install --no-root --with utils,server,dev --all-extras --compile

EXPOSE $PORT
#CMD ["bash"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]

################################
# PRODUCTION
# Final image used for runtime
################################
FROM python-base as production

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    apt-get clean

# copy in our built poetry + venv
COPY --from=builder-base $POETRY_HOME $POETRY_HOME
COPY --from=builder-base $VIRTUAL_ENV $VIRTUAL_ENV
# copy in our C dependencies. Remove if you don't need.
#COPY --from=builder-base /app/scripts/lib/*.so /usr/lib
# copy in pre-populated transformer cache. Remove if you don't need.
#COPY --from=builder-base $TRANSFORMERS_CACHE $TRANSFORMERS_CACHE

WORKDIR /app
COPY poetry.lock pyproject.toml ./
COPY . /app
#COPY my_awesome_ai_project/ my_awesome_ai_project/

EXPOSE $PORT
WORKDIR /app/litellm
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
