#!/usr/bin/env bash
set -euo pipefail

# --------- Defaults ---------
ENV_NAME="py38"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/content/micromamba}"
CHANNELS="${CHANNELS:-conda-forge}"
CWD=""
# Repeatable args -> arrays
declare -a PIP_GROUPS
declare -a RUN_ENTRIES

# --------- Help ---------
print_help() {
  cat << EOF
	Usage:
	  setup_py38_colab.sh [OPTIONS]

	Options (repeatable unless noted):
	  --env-name NAME         (single) Micromamba env name [default: py38]
	  --pip "PKG1 PKG2..."    Add a space-separated group of pip packages to install
	  --run "SCRIPT [ARGS]"   Run a Python file (with optional args) inside the env
	  --run-cmd "CODE"        Run an inline Python snippet inside the env
	  --cwd PATH              (single) cd into PATH before installing/running
	  -h, --help              Show this help and exit

	Examples:
	  setup_py38_colab.sh \
		--pip "numpy pandas" \
		--pip "scikit-learn shap" \
		--run "/content/train.py --epochs 5" \
		--run "/content/eval.py" \
		--run-cmd "import sys; print(sys.version)"
EOF
}

# --------- Parse args ---------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)     ENV_NAME="$2"; shift 2 ;;
    --pip)          PIP_GROUPS+=("$2"); shift 2 ;;
    --run)          RUN_ENTRIES+=("$2"); shift 2 ;;
    --cwd)          CWD="$2"; shift 2 ;;
    -h|--help)      print_help; exit 0 ;;
    *) echo "Unknown arg: $1"; print_help; exit 1 ;;
  esac
done

have() { command -v "$1" >/dev/null 2>&1; }
log()  { printf "\n[py38-setup] %s\n" "$*" >&2; }

# Optional working dir
if [[ -n "$CWD" ]]; then
  log "cd $CWD"
  cd "$CWD"
fi

# --------- Install micromamba if missing ---------
if ! have micromamba; then
  log "Installing micromamba under $MAMBA_ROOT_PREFIX ..."
  mkdir -p "$MAMBA_ROOT_PREFIX"
  pushd /tmp >/dev/null
  URL_BASE="https://micro.mamba.pm/api/micromamba/linux-64/latest"
  if have curl; then curl -Ls "$URL_BASE" -o micromamba.tar.bz2; else wget -qO micromamba.tar.bz2 "$URL_BASE"; fi
  tar -xjf micromamba.tar.bz2
  mkdir -p "$MAMBA_ROOT_PREFIX/bin"
  cp -f bin/micromamba "$MAMBA_ROOT_PREFIX/bin/micromamba"
  popd >/dev/null
fi
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

# --------- Create/verify env ---------
if ! micromamba env list | grep -E "^[[:space:]]*$ENV_NAME[[:space:]]" >/dev/null 2>&1; then
  log "Creating env '$ENV_NAME' with Python 3.8 ..."
  micromamba create -y -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" -c "$CHANNELS" python=3.8 pip
else
  log "Env '$ENV_NAME' already exists."
fi

log "Python version check:"
micromamba run -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" python - <<'PY'
import sys; print(sys.version)
PY

# --------- Install packages ---------
if ((${#PIP_GROUPS[@]})); then
  log "Upgrading pip..."
  micromamba run -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" python -m pip install -U pip
  for group in "${PIP_GROUPS[@]}"; do
    log "pip install $group"
    # shellcheck disable=SC2086
    micromamba run -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" python -m pip install $group
  done
fi

# --------- Execute runs (files then snippets) ---------

log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
log "Finish setup, start running the provided scripts"
log "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

if ((${#RUN_ENTRIES[@]})); then
  for entry in "${RUN_ENTRIES[@]}"; do
    log "Running: python $entry"
    # Word-splitting here is intentional to allow args, e.g. "/path/script.py --flag 1"
    # shellcheck disable=SC2086
    micromamba run -r "$MAMBA_ROOT_PREFIX" -n "$ENV_NAME" python $entry
  done
fi

if ! ((${#RUN_ENTRIES[@]})); then
  log "No --run provided. Environment is ready."
fi

log "Done."