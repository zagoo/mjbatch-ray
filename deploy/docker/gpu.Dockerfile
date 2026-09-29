# RaySimBatch CUDA inference image (PolicyInferenceService). Same release lock as the
# CPU image; CUDA runtime base pinned by digest at release time; onnxruntime-gpu and
# CUDA torch wheels replace the CPU builds. NOT built or tested on the qualification
# workstation (no CUDA device): see docs/qualification/report.md (BLOCKED).
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_IMAGE} AS build
ARG UV_VERSION=0.12.19
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3-pip \
    build-essential git ca-certificates && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m pip install --break-system-packages --no-cache-dir "uv==${UV_VERSION}"
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
COPY native native
COPY src src
COPY third_party/mjbatch third_party/mjbatch
RUN uv sync --frozen --no-dev --no-editable --extra torch --python python3.12 \
 && uv pip install --python .venv/bin/python "onnxruntime-gpu==1.30.0" \
 && uv pip uninstall --python .venv/bin/python onnxruntime

FROM ${CUDA_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 libgomp1 procps \
 && rm -rf /var/lib/apt/lists/* && useradd --uid 1000 --create-home ray
COPY --from=build /src/.venv /src/.venv
ENV PATH=/src/.venv/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
USER 1000
