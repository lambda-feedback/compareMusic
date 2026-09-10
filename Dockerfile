FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.11 AS builder

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock ./

RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --without dev --no-root

FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.11

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Precompile python files for faster startup
RUN python -m compileall -q .

# Copy the evaluation function to the app directory
COPY evaluation_function ./evaluation_function

# Command to start the evaluation function with
ENV FUNCTION_COMMAND="python"

# Args to start the evaluation function with
ENV FUNCTION_ARGS="-m,evaluation_function.main"

# The transport to use for the RPC server
ENV FUNCTION_INTERFACE="rpc"
ENV FUNCTION_RPC_TRANSPORT="stdio"

# The worker pulls in torch on its first request; on a small (1024 MB) Lambda
# that cold-start import runs ~30-40s. Give shimmy room to wait for it instead
# of killing the half-booted worker at the 30s default. Keep these below the
# Lambda function timeout (currently 175s).
ENV FUNCTION_WORKER_START_TIMEOUT="150s"
ENV FUNCTION_WORKER_SEND_TIMEOUT="150s"

ENV LOG_LEVEL="debug"
