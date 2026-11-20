# Copy to configs/env.sh (or any file), edit, then `source` it before running a script.
# Every path defaults to a location inside the package; nothing outside it is needed for the CPU regeneration.

export VTAX_ROOT="${VTAX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # package root (scripts/, outputs/, data/)
export VTAX_REGEN="${VTAX_REGEN:-$VTAX_ROOT/regenerated}"                       # regenerated tables/ and figures/
export VTAX_DATA_ROOT="${VTAX_DATA_ROOT:-$VTAX_ROOT/external}"                  # datasets/ and models/ (GPU / API runs only)
export VTAX_SCRATCH="${VTAX_SCRATCH:-/tmp/vtax}"                                # scratch space of the GPU / API runners

# GPU / API runs only: Hugging Face ids of two public benchmarks (see docs/REPRODUCE.md, "Data").
export CVBENCH_HF_ID="${CVBENCH_HF_ID:-}"
export VSIBENCH_HF_ID="${VSIBENCH_HF_ID:-}"

# Closed-model and judge calls (API runs only): any OpenAI-compatible chat-completions endpoint.
export OPENAI_COMPAT_BASE_URL="${OPENAI_COMPAT_BASE_URL:-}"   # e.g. https://<host>/v1
# export OPENAI_COMPAT_API_KEY=...                            # set in your shell; never commit it
# export OPENAI_COMPAT_API_KEY_ALT=...                        # optional second key, used by judges run with --key alt

# Settings used for the GPU runs.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
