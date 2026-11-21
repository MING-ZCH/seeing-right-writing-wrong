# Copy to configs/env.sh (or any file), edit, then `source` it before running a script from the repository root.

export VTAX_ROOT="${VTAX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # repository root (srww/, experiments/, data/, outputs/)
export VTAX_DATA_ROOT="${VTAX_DATA_ROOT:-$VTAX_ROOT/external}"                  # datasets/ and models/ (GPU / API runs)
export VTAX_SCRATCH="${VTAX_SCRATCH:-/tmp/vtax}"                                # scratch space of the GPU / API runners
export PYTHONPATH="$VTAX_ROOT${PYTHONPATH:+:$PYTHONPATH}"                       # lets `python experiments/...py` find srww/

# Closed-model and judge calls (API runs only): any OpenAI-compatible chat-completions endpoint.
export OPENAI_COMPAT_BASE_URL="${OPENAI_COMPAT_BASE_URL:-}"   # e.g. https://<host>/v1
# export OPENAI_COMPAT_API_KEY=...                            # set in your shell; never commit it
# export OPENAI_COMPAT_API_KEY_ALT=...                        # optional second key, used by judges run with --key alt

# Optional: LoRA pilot checkpoints (default /dev/shm/e63_<arm>).
# export E63_CKPT_MATCH=...
# export E63_CKPT_CF=...

# Settings used for the GPU runs.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
