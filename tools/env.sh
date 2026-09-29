# Source from the repository root: `source tools/env.sh`.
RSB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export RSB_ROOT
# The project environment (created by `uv sync`) first, so documented `python ...`
# commands run with RaySimBatch installed; then the pinned tools (uv, kind, helm, kubectl).
export PATH="$RSB_ROOT/.venv/bin:$RSB_ROOT/.tools/venv/bin:$RSB_ROOT/.tools/bin:$PATH"
export UV_PYTHON="python3.12"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Builds must leave no artifacts in third_party/mjbatch (tools/check_mjbatch_clean.py).
export PYTHONDONTWRITEBYTECODE=1
# Keep Helm and kind state inside the project (never the user's global config).
export HELM_CONFIG_HOME="$RSB_ROOT/.tools/helm/config" HELM_CACHE_HOME="$RSB_ROOT/.tools/helm/cache" HELM_DATA_HOME="$RSB_ROOT/.tools/helm/data"
export KUBECONFIG="$RSB_ROOT/.tools/kubeconfig"
