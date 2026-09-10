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

# The transport to use for the RPC server. Use the unix-socket transport
# ("ipc"), not "stdio": under stdio the worker's stdout *is* the RPC wire, and
# this function's heavy import stack (basic-pitch / onnxruntime / numba) writes
# to stdout outside any redirect guard, which would corrupt the stream.
#
# FUNCTION_* configures shimmy (the Go side). The Python worker picks its
# transport separately from EVAL_* (see lf_toolkit/io/serve.py, which defaults
# EVAL_RPC_TRANSPORT to "stdio"), so set those explicitly too - otherwise the
# worker comes up as a StdioServer, never binds /tmp/eval.sock, and every
# request fails with "error sending rpc request: EOF".
ENV FUNCTION_INTERFACE="rpc"
ENV FUNCTION_RPC_TRANSPORT="ipc"
ENV EVAL_IO="rpc"
ENV EVAL_RPC_TRANSPORT="ipc"
ENV EVAL_RPC_IPC_ENDPOINT="/tmp/eval.sock"

# The worker pulls in a large ML stack on its first request; on a small
# (1024 MB) Lambda that cold-start import runs ~30-40s. Give shimmy room to
# wait for it instead of killing the half-booted worker at the 30s default.
# Keep these below the Lambda function timeout (currently 175s).
ENV FUNCTION_WORKER_START_TIMEOUT="150s"
ENV FUNCTION_WORKER_SEND_TIMEOUT="150s"

ENV LOG_LEVEL="debug"
