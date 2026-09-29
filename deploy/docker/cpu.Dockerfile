# RaySimBatch CPU simulation image. Built from one release lock (uv.lock + pinned
# submodule + digests below): MJBatch is compiled from the pinned upstream submodule,
# unmodified; RaySimBatch's native kernels are compiled from native/. No runtime pip
# installation happens in production.
ARG PYTHON_IMAGE=python:3.12.8-slim-bookworm@sha256:2199a62885a12290dc9c5be3ca0681d367576ab7bf037da120e564723292a2f0
FROM ${PYTHON_IMAGE} AS build
ARG UV_VERSION=0.12.19
ARG EXTRAS=""
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never UV_PYTHON=/usr/local/bin/python3.12
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "uv==${UV_VERSION}"
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
COPY native native
COPY src src
COPY third_party/mjbatch third_party/mjbatch
RUN uv sync --frozen --no-dev --no-editable ${EXTRAS:+--extra ${EXTRAS}} \
 && .venv/bin/python -m raysimbatch.cli compat > /src/compat.json \
 && test -z "$(cd third_party/mjbatch && find . -name build -maxdepth 1)"

# Test stage: same build plus dev dependencies and the test suites, for Linux-only
# qualification (affinity, sanitizers) inside the container. Not a production image.
FROM build AS test
RUN uv sync --frozen --no-editable
COPY tests tests
ENV PATH=/src/.venv/bin:$PATH

# Sanitizer stage: MJBatch (unmodified sources) and the native kernels rebuilt with
# sanitizer *build flags only*; run with the runtime preloaded (see
# qualification/native_sanitizers.sh).
FROM test AS sanitize
ARG SAN=address,undefined
RUN uv sync --frozen --no-editable --reinstall-package mjbatch --reinstall-package raysimbatch \
      --config-settings-package "mjbatch:cmake.define.CMAKE_CXX_FLAGS=-fsanitize=${SAN} -fno-omit-frame-pointer -g -O1" \
      --config-settings-package "mjbatch:cmake.define.CMAKE_SHARED_LINKER_FLAGS=-fsanitize=${SAN}" \
      --config-settings-package "raysimbatch:cmake.define.RSB_SANITIZE=${SAN}"

# Production runtime stage. It is deliberately the LAST stage so that a plain
# `docker build` (no --target) yields the production image, never a test or
# sanitizer image; deploy/docker/build.sh also passes --target runtime explicitly.
FROM ${PYTHON_IMAGE} AS runtime
ARG GIT_SHA=unknown
ARG MJBATCH_SHA=unknown
LABEL org.opencontainers.image.title="raysimbatch-cpu" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      io.raysimbatch.mjbatch.revision="${MJBATCH_SHA}"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 procps \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 1000 --create-home ray \
 && mkdir -p /var/lib/ray-spill /var/lib/raysimbatch /etc/raysimbatch \
 && chown ray:ray /var/lib/ray-spill /var/lib/raysimbatch
COPY --from=build /src/.venv /src/.venv
COPY --from=build /src/compat.json /etc/raysimbatch/compat.json
ENV PATH=/src/.venv/bin:$PATH \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    RSB_GIT_SHA=${GIT_SHA} RSB_MJBATCH_SHA=${MJBATCH_SHA}
USER 1000
WORKDIR /home/ray
