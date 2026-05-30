#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_gpu.sh --mode reml|lasso|gwas [launcher args...] [-- pipeline args...]

Modes:
  reml      REML via run_reml_pipeline.py; supports single-GRM, multi-GRM, and dense+sparse mixed input
  lasso     Sparse REML + LASSO via run_sparse_reml_pipeline.py
  gwas      Continuous-trait marginal GWAS via run_gwas_pipeline.py

Launcher args:
  --bed-prefix PATH
  --pgen-prefix PATH
  --rare-bed-prefix PATH
  --rare-pgen-prefix PATH
  --pheno-txt PATH
  --covar-txt PATH
  --prediction-bed-prefix PATH
  --prediction-pgen-prefix PATH
  --prediction-covar-txt PATH
  --keep-path PATH
  --gpu-budget-gb NUM
  --ring-depth NUM
  --out-prefix PATH
  --log-label NAME
  --python-bin PATH

Environment overrides:
  CUDA_VISIBLE_DEVICES, OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS
  PYTHON_BIN

Examples:
  ./run_gpu.sh --mode reml --bed-prefix /path/to/data --pheno-txt /path/to/pheno.txt --covar-txt /path/to/covar.txt
  ./run_gpu.sh --mode reml --bed-prefix /path/to/data --pheno-txt /path/to/pheno.txt --covar-txt /path/to/covar.txt --compute-effects --out-prefix ./logs/reml_effects
  ./run_gpu.sh --mode reml --bed-prefix /path/to/train --pheno-txt /path/to/pheno.txt --covar-txt /path/to/train.covar --prediction-bed-prefix /path/to/test --prediction-covar-txt /path/to/test.covar --out-prefix ./logs/reml_pred
  ./run_gpu.sh --mode reml --bed-prefix /DATA/jiaxin/dense --rare-bed-prefix /DATA/jiaxin/rare
  ./run_gpu.sh --mode lasso --bed-prefix /path/to/data --pheno-txt /path/to/pheno.txt --covar-txt /path/to/covar.txt --out-prefix ./logs/lasso_run
  ./run_gpu.sh --mode gwas --bed-prefix /path/to/data --pheno-txt /path/to/pheno.txt --covar-txt /path/to/covar.txt --out-prefix ./logs/gwas_run
EOF
}

MODE="${MODE:-reml}"
BED_PREFIX=""
PGEN_PREFIX=""
RARE_BED_PREFIX=""
RARE_PGEN_PREFIX=""
PHENO_TXT=""
COVAR_TXT=""
KEEP_PATH=""
GPU_BUDGET_GB="${GPU_BUDGET_GB:-40}"
RING_DEPTH="${RING_DEPTH:-0}"
OUT_PREFIX=""
LOG_LABEL="${LOG_LABEL:-}"
EXTRA_ARGS=()

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    --bed-prefix)
      BED_PREFIX="${2:-}"
      shift 2
      ;;
    --bed-prefix=*)
      BED_PREFIX="${1#--bed-prefix=}"
      shift
      ;;
    --pgen-prefix)
      PGEN_PREFIX="${2:-}"
      shift 2
      ;;
    --pgen-prefix=*)
      PGEN_PREFIX="${1#--pgen-prefix=}"
      shift
      ;;
    --rare-bed-prefix)
      RARE_BED_PREFIX="${2:-}"
      shift 2
      ;;
    --rare-bed-prefix=*)
      RARE_BED_PREFIX="${1#--rare-bed-prefix=}"
      shift
      ;;
    --rare-pgen-prefix)
      RARE_PGEN_PREFIX="${2:-}"
      shift 2
      ;;
    --rare-pgen-prefix=*)
      RARE_PGEN_PREFIX="${1#--rare-pgen-prefix=}"
      shift
      ;;
    --pheno-txt)
      PHENO_TXT="${2:-}"
      shift 2
      ;;
    --pheno-txt=*)
      PHENO_TXT="${1#--pheno-txt=}"
      shift
      ;;
    --covar-txt)
      COVAR_TXT="${2:-}"
      shift 2
      ;;
    --covar-txt=*)
      COVAR_TXT="${1#--covar-txt=}"
      shift
      ;;
    --keep-path)
      KEEP_PATH="${2:-}"
      shift 2
      ;;
    --keep-path=*)
      KEEP_PATH="${1#--keep-path=}"
      shift
      ;;
    --gpu-budget-gb)
      GPU_BUDGET_GB="${2:-}"
      shift 2
      ;;
    --gpu-budget-gb=*)
      GPU_BUDGET_GB="${1#--gpu-budget-gb=}"
      shift
      ;;
    --ring-depth)
      RING_DEPTH="${2:-}"
      shift 2
      ;;
    --ring-depth=*)
      RING_DEPTH="${1#--ring-depth=}"
      shift
      ;;
    --out-prefix)
      OUT_PREFIX="${2:-}"
      shift 2
      ;;
    --out-prefix=*)
      OUT_PREFIX="${1#--out-prefix=}"
      shift
      ;;
    --log-label)
      LOG_LABEL="${2:-}"
      shift 2
      ;;
    --log-label=*)
      LOG_LABEL="${1#--log-label=}"
      shift
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --python-bin=*)
      PYTHON_BIN="${1#--python-bin=}"
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "$MODE" in
  reml|lasso|gwas) ;;
  *)
    echo "[ERROR] unknown mode: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac

echo "Start: $(date)"
echo "Host : $(hostname)"
echo "Mode : $MODE"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs"

export PATH=/usr/local/cuda-12.6/bin:$PATH
# Disable JAX's large upfront GPU reservation so observed device memory stays
# closer to the planner's working-set estimate and historical benchmark runs.
# We intentionally do not force XLA_PYTHON_CLIENT_ALLOCATOR=platform here:
# PREALLOCATE=false was sufficient in prior runs near the GPU budget, while
# forcing the platform allocator can change allocation behavior and throughput.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export VERBOSE="${VERBOSE:-1}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[ERROR] no usable Python runtime found. Activate an environment first, or set PYTHON_BIN." >&2
    exit 1
  fi
fi
export PYTHON_BIN
export PYTHONPATH="$(dirname "$REPO_ROOT"):${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_STAMP="$(date +%F_%H%M%S)"

sanitize_label() {
  local raw="$1"
  raw="${raw// /_}"
  raw="${raw//\//_}"
  raw="${raw//,/__}"
  raw="$(printf '%s' "$raw" | tr -cd '[:alnum:]._+-')"
  printf '%s' "${raw:-${MODE}_${RUN_STAMP}}"
}

if [[ -z "$OUT_PREFIX" ]]; then
  if [[ -n "$LOG_LABEL" ]]; then
    OUT_PREFIX="$REPO_ROOT/logs/$(sanitize_label "$LOG_LABEL")_${RUN_STAMP}"
  else
    OUT_PREFIX="$REPO_ROOT/logs/${MODE}_${RUN_STAMP}"
  fi
fi

RUN_LABEL="$(sanitize_label "$(basename "$OUT_PREFIX")")"
LOG_BASE="$REPO_ROOT/logs/$RUN_LABEL"
LOG="${LOG_BASE}.log"
CPU_MON_LOG="${LOG_BASE}.cpu.csv"

log_msg() {
  echo "$1" | tee -a "$LOG"
}

cleanup_monitors() {
  if [[ -n "${GPU_MON_PID:-}" ]]; then
    kill "$GPU_MON_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CPU_MON_PID:-}" ]]; then
    kill "$CPU_MON_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup_monitors EXIT

GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
GPU_ID="${GPU_ID:-0}"
GPU_MON_LOG="${LOG_BASE}.gpu.csv"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "timestamp,utilization.gpu,memory.used,memory.total" > "$GPU_MON_LOG"
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 1 >> "$GPU_MON_LOG" &
  GPU_MON_PID=$!
  log_msg "[INFO] gpu monitor -> $GPU_MON_LOG (gpu_id=$GPU_ID)"
else
  log_msg "[WARN] nvidia-smi not found; skip GPU monitor"
fi

echo "timestamp,rss_kb,hwm_kb,vmsize_kb,threads" > "$CPU_MON_LOG"

GPU_BUDGET_ARGS=()
if [[ "${GPU_BUDGET_GB}" != "0" ]]; then
  GPU_BUDGET_ARGS=(--gpu-budget-gb "$GPU_BUDGET_GB")
fi

RING_DEPTH_ARGS=()
if [[ "${RING_DEPTH}" != "0" ]]; then
  RING_DEPTH_ARGS=(--ring-depth "$RING_DEPTH")
fi

ENTRYPOINT=""
PIPELINE_ARGS=()

if [[ -z "$PHENO_TXT" ]]; then
  echo "[ERROR] PHENO_TXT is required" >&2
  exit 1
fi

case "$MODE" in
  reml)
    ENTRYPOINT="$REPO_ROOT/run_reml_pipeline.py"
    INPUT_ARGS=()
    [[ -n "$BED_PREFIX" ]] && INPUT_ARGS+=(--bed-prefix "$BED_PREFIX")
    [[ -n "$PGEN_PREFIX" ]] && INPUT_ARGS+=(--pgen-prefix "$PGEN_PREFIX")
    [[ -n "$RARE_BED_PREFIX" ]] && INPUT_ARGS+=(--rare-bed-prefix "$RARE_BED_PREFIX")
    [[ -n "$RARE_PGEN_PREFIX" ]] && INPUT_ARGS+=(--rare-pgen-prefix "$RARE_PGEN_PREFIX")
    if [[ "${#INPUT_ARGS[@]}" -eq 0 ]]; then
      echo "[ERROR] reml mode requires at least one of BED_PREFIX / PGEN_PREFIX / RARE_BED_PREFIX / RARE_PGEN_PREFIX" >&2
      exit 1
    fi
    PIPELINE_ARGS=(
      "${INPUT_ARGS[@]}"
      --pheno-txt "$PHENO_TXT"
      --covar-txt "$COVAR_TXT"
    )
    [[ -n "$KEEP_PATH" ]] && PIPELINE_ARGS+=(--keep-path "$KEEP_PATH")
    [[ -n "$OUT_PREFIX" ]] && PIPELINE_ARGS+=(--out-prefix "$OUT_PREFIX")
    PIPELINE_ARGS+=("${GPU_BUDGET_ARGS[@]}" "${RING_DEPTH_ARGS[@]}")
    ;;
  lasso)
    ENTRYPOINT="$REPO_ROOT/run_sparse_reml_pipeline.py"
    input_count=0
    [[ -n "$BED_PREFIX" ]] && input_count=$((input_count + 1))
    [[ -n "$PGEN_PREFIX" ]] && input_count=$((input_count + 1))
    if [[ "$input_count" -ne 1 ]]; then
      echo "[ERROR] lasso mode requires exactly one of BED_PREFIX / PGEN_PREFIX" >&2
      exit 1
    fi
    INPUT_ARGS=()
    [[ -n "$BED_PREFIX" ]] && INPUT_ARGS+=(--bed-prefix "$BED_PREFIX")
    [[ -n "$PGEN_PREFIX" ]] && INPUT_ARGS+=(--pgen-prefix "$PGEN_PREFIX")
    PIPELINE_ARGS=(
      "${INPUT_ARGS[@]}"
      --pheno-txt "$PHENO_TXT"
      --covar-txt "$COVAR_TXT"
      --out-prefix "$OUT_PREFIX"
      --verbose
    )
    [[ -n "$KEEP_PATH" ]] && PIPELINE_ARGS+=(--keep-path "$KEEP_PATH")
    PIPELINE_ARGS+=("${GPU_BUDGET_ARGS[@]}" "${RING_DEPTH_ARGS[@]}")
    ;;
  gwas)
    ENTRYPOINT="$REPO_ROOT/run_gwas_pipeline.py"
    input_count=0
    [[ -n "$BED_PREFIX" ]] && input_count=$((input_count + 1))
    [[ -n "$PGEN_PREFIX" ]] && input_count=$((input_count + 1))
    if [[ "$input_count" -ne 1 ]]; then
      echo "[ERROR] gwas mode requires exactly one of BED_PREFIX / PGEN_PREFIX" >&2
      exit 1
    fi
    if [[ -z "$OUT_PREFIX" ]]; then
      echo "[ERROR] gwas mode requires OUT_PREFIX" >&2
      exit 1
    fi
    INPUT_ARGS=()
    [[ -n "$BED_PREFIX" ]] && INPUT_ARGS+=(--bed-prefix "$BED_PREFIX")
    [[ -n "$PGEN_PREFIX" ]] && INPUT_ARGS+=(--pgen-prefix "$PGEN_PREFIX")
    PIPELINE_ARGS=(
      "${INPUT_ARGS[@]}"
      --pheno-txt "$PHENO_TXT"
      --out-prefix "$OUT_PREFIX"
      --verbose
    )
    [[ -n "$COVAR_TXT" ]] && PIPELINE_ARGS+=(--covar-txt "$COVAR_TXT")
    [[ -n "$KEEP_PATH" ]] && PIPELINE_ARGS+=(--keep-path "$KEEP_PATH")
    PIPELINE_ARGS+=("${RING_DEPTH_ARGS[@]}")
    ;;
esac

log_msg "[INFO] log -> $LOG"
log_msg "[INFO] entrypoint -> $ENTRYPOINT"
log_msg "[INFO] run label -> $RUN_LABEL"
log_msg "[INFO] OUT_PREFIX -> $OUT_PREFIX"
log_msg "[INFO] VERBOSE -> $VERBOSE"

"$PYTHON_BIN" -u "$ENTRYPOINT" \
  "${PIPELINE_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" > >(tee -a "$LOG") 2>&1 &
PY_PID=$!
log_msg "[INFO] cpu monitor -> $CPU_MON_LOG (pid=$PY_PID)"

(
  while kill -0 "$PY_PID" >/dev/null 2>&1; do
    ts="$(date '+%F %T')"
    status_path="/proc/$PY_PID/status"
    if [[ -r "$status_path" ]]; then
      rss_kb="$(awk '/^VmRSS:/ {print $2}' "$status_path")"
      hwm_kb="$(awk '/^VmHWM:/ {print $2}' "$status_path")"
      vmsize_kb="$(awk '/^VmSize:/ {print $2}' "$status_path")"
      threads="$(awk '/^Threads:/ {print $2}' "$status_path")"
      echo "$ts,${rss_kb:-},${hwm_kb:-},${vmsize_kb:-},${threads:-}" >> "$CPU_MON_LOG"
    fi
    sleep 1
  done
) &
CPU_MON_PID=$!

set +e
wait "$PY_PID"
PY_STATUS=$?
set -e

wait "$CPU_MON_PID" 2>/dev/null || true

CPU_PEAK_RSS_KB="$(awk -F, 'NR > 1 && $2 + 0 > m { m = $2 + 0 } END { print m + 0 }' "$CPU_MON_LOG")"
CPU_PEAK_HWM_KB="$(awk -F, 'NR > 1 && $3 + 0 > m { m = $3 + 0 } END { print m + 0 }' "$CPU_MON_LOG")"
log_msg "[INFO] cpu peak -> rss_kb=$CPU_PEAK_RSS_KB hwm_kb=$CPU_PEAK_HWM_KB"

echo "End time: $(date)"
echo "Log file: $LOG"
exit "$PY_STATUS"
