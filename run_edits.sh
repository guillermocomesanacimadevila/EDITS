#!/usr/bin/env bash
# run_edits.sh

# EDITS PIPELINE: 
# - Outputs under: outputs/<project>/<run_id>/
# - Phases:
#     phase1_pretraining/
#     phase2_data_prep/
#     phase3_supervised_classification/
#     phase4_mistake_analysis/
#     phase5_grad_cam/
# - Centralised:
#     config/, logs/, models/, figures/, metrics/, report.html

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

center_text() {
  echo -e "$1"
}

IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
  IS_WSL=true
fi

IS_VAST=false
if [[ "$EDITS_PLATFORM" == "vast" ]]; then
  IS_VAST=true
else
  if [[ $EUID -eq 0 ]] && command -v apt-get &>/dev/null; then
    if hostname | grep -qE '^C\.[0-9]+' || grep -q 'developer.download.nvidia.com/compute/cuda' /etc/apt/sources.list* 2>/dev/null; then
      IS_VAST=true
    fi
  fi
fi

REQUIRED_COMMANDS=(wget python3 pip conda)
for cmd in "${REQUIRED_COMMANDS[@]}"; do
  if ! command -v "$cmd" &> /dev/null; then
    echo -e "${RED}❌ Required command '$cmd' not found. Please install it before running this script.${NC}"
    exit 1
  fi
done

REQUIRED_SPACE_MB=500
AVAILABLE_SPACE_KB=$(df "$HOME" | tail -1 | awk '{print $4}')
AVAILABLE_SPACE_MB=$((AVAILABLE_SPACE_KB / 1024))
if (( AVAILABLE_SPACE_MB < REQUIRED_SPACE_MB )); then
  echo -e "${RED}❌ Not enough disk space: ${AVAILABLE_SPACE_MB}MB available, ${REQUIRED_SPACE_MB}MB needed in $HOME.${NC}"
  exit 1
fi

PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
if [[ "$PYTHON_MAJOR" != "3" ]]; then
  echo -e "${RED}❌ Python 3 is required. Detected major version: $PYTHON_MAJOR${NC}"
  exit 1
fi

echo -e "${BLUE}🔧 Checking and installing system requirements...${NC}"

install_linux_tools() {
  local pkgs=(python3 python3-pip wget git time fontconfig fonts-dejavu-core coreutils findutils)
  if command -v sudo &>/dev/null && [[ $EUID -ne 0 ]]; then
    sudo apt-get update -qq || true
    sudo apt-get install -y "${pkgs[@]}" fzf || true
  else
    apt-get update -qq || true
    apt-get install -y "${pkgs[@]}" fzf || true
  fi

  if ! command -v fzf >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  'fzf' not available via apt — installing from source...${NC}"
    tmpdir="$(mktemp -d)"
    git clone --depth 1 https://github.com/junegunn/fzf.git "$tmpdir/fzf" >/dev/null 2>&1 || true
    bash "$tmpdir/fzf/install" --bin --no-update-rc --no-key-bindings --no-completion >/dev/null 2>&1 || true
    if [[ -f "$tmpdir/fzf/bin/fzf" ]]; then
      install -m 0755 "$tmpdir/fzf/bin/fzf" /usr/local/bin/fzf 2>/dev/null || cp "$tmpdir/fzf/bin/fzf" /usr/bin/fzf 2>/dev/null || true
    fi
    rm -rf "$tmpdir"
  fi

  export TERM=${TERM:-xterm-256color}
}

OS_TYPE="$(uname)"
if [[ "$OS_TYPE" == "Linux" ]] && command -v apt-get &>/dev/null; then
  if $IS_VAST; then
    echo -e "${BLUE}➡️  Linux + apt detected (Vast.ai mode).${NC}"
  else
    echo -e "${BLUE}➡️  Linux + apt detected.${NC}"
  fi
  install_linux_tools
  if $IS_WSL && ! command -v wslview &>/dev/null; then
    apt-get install -y wslu >/dev/null 2>&1 || true
  fi
elif [[ "$OS_TYPE" == "Darwin" ]]; then
  echo -e "${BLUE}➡️  macOS detected. Using Homebrew if available.${NC}"
  if ! command -v brew >/dev/null 2>&1; then
    echo -e "${YELLOW}ℹ️  Please install Homebrew: https://brew.sh${NC}"
  else
    for brewdep in python3 wget git fzf fontconfig; do
      brew list "$brewdep" >/dev/null 2>&1 || brew install "$brewdep"
    done
    if ! command -v gtime >/dev/null 2>&1; then
      echo -e "${YELLOW}ℹ️  For detailed timing metrics, install GNU time: 'brew install coreutils' (provides 'gtime').${NC}"
    fi
  fi
fi

run_phase() {
  local STEP="$1"
  local LOG="$2"
  local DISK_DIR="$3"
  local CSV_FILE="$4"
  shift 4
  local CMD=("$@")

  mkdir -p "$(dirname "$LOG")"

  local time_cmd=""
  if command -v gtime >/dev/null 2>&1; then
    time_cmd="gtime -v"
  elif command -v /usr/bin/time >/dev/null 2>&1 && /usr/bin/time -v bash -c ":" 2>/dev/null; then
    time_cmd="/usr/bin/time -v"
  elif command -v /usr/bin/time >/dev/null 2>&1; then
    time_cmd="/usr/bin/time -l"
  fi

  echo ""
  echo -e "${BLUE}${STEP}${NC}"
  echo "  logs -> $LOG"

  if [[ -n "$time_cmd" ]]; then
    ( $time_cmd "${CMD[@]}" >"$LOG" 2>&1 ) &
  else
    ( "${CMD[@]}" >"$LOG" 2>&1 ) &
  fi
  local pid=$!

  local spin='-\|/'
  local i=0
  local progress=0
  local width=30

  while kill -0 "$pid" 2>/dev/null; do
    i=$(( (i+1) % 4 ))
    if (( progress < 99 )); then
      progress=$((progress + 1))
    fi
    local filled=$((progress * width / 100))
    local empty=$((width - filled))
    printf "\r  [%c] [%-*s] %3d%%" \
      "${spin:$i:1}" \
      "$width" "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
      "$progress"
    sleep 0.2
  done

  wait "$pid"
  local STATUS=$?

  local ELAPSED_SEC="NA"
  local PEAK_RAM_MB="NA"

  if [[ $STATUS -eq 0 ]]; then
    local filled=$width
    progress=100
    printf "\r  [✔] [%-*s] %3d%%\n" \
      "$width" "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
      "$progress"
  else
    printf "\r  [✖] Phase '%s' failed. See log: %s\n" "$STEP" "$LOG"
  fi

  if grep -q "Elapsed (wall clock) time" "$LOG" 2>/dev/null; then
    local ELAPSED
    ELAPSED=$(grep -E "Elapsed \(wall clock\) time" "$LOG" | tail -1 | awk '{print $8}')
    local h=0 m=0 s=0
    if [[ "$ELAPSED" == *:*:* ]]; then IFS=: read -r h m s <<< "$ELAPSED"
    elif [[ "$ELAPSED" == *:* ]]; then IFS=: read -r m s <<< "$ELAPSED"
    else s="$ELAPSED"; fi
    s=${s%%.*}; m=${m:-0}; h=${h:-0}
    ELAPSED_SEC=$((10#$h*3600 + 10#$m*60 + 10#$s))
  fi

  if grep -qi "maximum resident set size" "$LOG" 2>/dev/null; then
    local PEAK_RAM_KB
    PEAK_RAM_KB=$(grep -i "maximum resident set size" "$LOG" | tail -1 | awk '{print $1}')
    [[ -n "$PEAK_RAM_KB" ]] && PEAK_RAM_MB=$((PEAK_RAM_KB / 1024))
  fi

  local DISK_MB="NA"
  if [[ -d "$DISK_DIR" ]]; then
    DISK_MB=$(du -sm "$DISK_DIR" 2>/dev/null | awk '{print $1}')
  fi

  if [[ -n "$CSV_FILE" ]]; then
    echo "$STEP,$ELAPSED_SEC,${PEAK_RAM_MB:-NA},${DISK_MB:-NA}" >> "$CSV_FILE"
  fi

  if [[ $STATUS -ne 0 ]]; then
    echo "── Last 80 lines of $LOG ──"
    tail -n 80 "$LOG" || true
    exit $STATUS
  else
    echo -e "${GREEN}✅ ${STEP} completed.${NC}"
  fi
}

open_html_report() {
  local html_path="$1"
  if [ -f "$html_path" ]; then
    if $IS_WSL && command -v wslview &> /dev/null; then
      wslview "$html_path"
    elif command -v xdg-open &> /dev/null; then
      xdg-open "$html_path"
    elif command -v open &> /dev/null; then
      open "$html_path"
    elif command -v cygstart &> /dev/null; then
      cygstart "$html_path"
    else
      echo -e "${YELLOW}⚠️  Could not auto-open. Open manually:${NC} file://$html_path"
    fi
  else
    echo -e "${RED}❌ HTML report not found: $html_path${NC}"
  fi
}

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  echo -e "${YELLOW}EDITS PIPELINE (single-run)${NC}"
  echo "Usage: bash $0"
  exit 0
fi

trap 'echo -e "\n${RED}⚡️ Script interrupted by user. Exiting!${NC}"; exit 1' SIGINT

ENV_NAME="edits-env"
ENV_YML="env/environment.yml"

if ! command -v conda &> /dev/null; then
  echo -e "${YELLOW}🔄 Conda not found. Installing Miniconda...${NC}"
  wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$HOME/miniconda"
  export PATH="$HOME/miniconda/bin:$PATH"
  source "$HOME/miniconda/etc/profile.d/conda.sh"
  command -v conda &>/dev/null || { echo -e "${RED}❌ Miniconda install failed.${NC}"; exit 1; }
  echo -e "${GREEN}✅ Miniconda installed.${NC}"
else
  eval "$(conda shell.bash hook)"
fi

if [ ! -f "$ENV_YML" ]; then
  echo -e "${RED}❌ $ENV_YML not found! Cannot create conda env.${NC}"
  exit 1
fi

if ! conda env list | grep -qw "$ENV_NAME"; then
  echo -e "${YELLOW}🔧 Creating Conda env '$ENV_NAME' from $ENV_YML...${NC}"
  conda env create -f "$ENV_YML" -n "$ENV_NAME" || { echo -e "${RED}❌ Failed to create env.${NC}"; exit 1; }
  echo -e "${GREEN}✅ Conda environment '$ENV_NAME' created.${NC}"
fi

echo -e "${GREEN}🔄 Activating '$ENV_NAME'...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo -e "${RED}❌ Failed to activate '$ENV_NAME'!${NC}"; exit 1; }

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p ~/.config/matplotlib
cat > ~/.config/matplotlib/matplotlibrc <<'RC'
font.family: sans-serif
font.sans-serif: DejaVu Sans
RC
mkdir -p ~/.cache/matplotlib
chmod 700 ~/.cache/matplotlib

python3 -c "import torch, numpy, matplotlib" 2>/dev/null || {
  echo -e "${RED}❌ Required Python packages missing (torch, numpy, matplotlib).${NC}"
  exit 1
}

echo -e "${YELLOW}🔗 Ensuring TAP/tarrow package is installed (editable)...${NC}"
if [ -d "TAP/tarrow" ] && [ -f "TAP/tarrow/setup.py" ]; then
  pip show tarrow > /dev/null 2>&1 || pip install -e TAP/tarrow || { echo -e "${RED}❌ Failed to install TAP/tarrow.${NC}"; exit 1; }
else
  echo -e "${RED}❌ TAP/tarrow directory or setup.py not found!${NC}"
  exit 1
fi

echo -e "${YELLOW}🔢 Environment Versions:${NC}"
echo -n "Python: "; python3 --version
echo -n "Conda: "; conda --version
echo -n "PyTorch: "; python3 -c 'import torch; print(torch.__version__)' || echo "N/A"
echo -n "Numpy: "; python3 -c 'import numpy; print(numpy.__version__)' || echo "N/A"
echo "----------------------------"

select_file() {
  local prompt="$1"
  local start_dir="$2"
  local file
  if command -v fzf &> /dev/null; then
    export TERM=${TERM:-xterm-256color}
    file=$(find "$start_dir" \( -name "*.tif" -o -name "*.tiff" \) -type f 2>/dev/null | fzf --prompt="$prompt " --height=15 --border)
    [ -z "$file" ] && { echo -e "${RED}❌ No file selected! Exiting.${NC}"; exit 1; }
    [[ "$file" == "$PWD"* ]] && file="."${file#$PWD}
    echo "$file"
  else
    echo -e "${YELLOW}Available files in $start_dir:${NC}"
    find "$start_dir" \( -name "*.tif" -o -name "*.tiff" \) -type f | nl
    read -rp "$prompt (copy-paste or type relative path): " file
    [ ! -f "$file" ] && { echo -e "${RED}❌ File not found: $file${NC}" ; exit 1; }
    echo "$file"
  fi
}

select_pretrained_model() {
  local root="${1:-.}"
  local tmp_list
  tmp_list=$(mktemp)

  while IFS= read -r d; do
    echo "[FOLDER]  $d" >> "$tmp_list"
  done < <(find "$root" -type f -name "model_kwargs.yaml" 2>/dev/null | sed 's:/model_kwargs.yaml$::' | sort -u)

  while IFS= read -r f; do
    echo "[WEIGHTS] $f" >> "$tmp_list"
  done < <(find "$root" -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.pkl" \) 2>/dev/null | sort -u)

  if ! [ -s "$tmp_list" ]; then
    echo -e "${RED}❌ No pre-trained models found under '$root'.${NC}"
    rm -f "$tmp_list"
    return 1
  fi

  local choice
  if command -v fzf >/dev/null 2>&1; then
    export TERM=${TERM:-xterm-256color}
    choice=$(fzf --prompt="📦 Select pre-trained TAP model (folder or weights): " --height=18 --border < "$tmp_list")
  else
    echo -e "${YELLOW}Available pre-trained models:${NC}"
    nl -ba "$tmp_list"
    read -rp "Enter the line number: " idx
    choice=$(sed -n "${idx}p" "$tmp_list")
  fi
  rm -f "$tmp_list"

  if [ -z "$choice" ]; then
    echo -e "${RED}❌ No selection made.${NC}"
    return 1
  fi

  local path="${choice#*] }"
  if [[ "$choice" == "[FOLDER]"* ]]; then
    echo "$path"
  else
    dirname "$path"
  fi
}

center_text "${BLUE}🔬 EDITS ML Pipeline Setup (Single Run)${NC}"
echo -e "${YELLOW}ℹ️  Select files interactively below (relative paths preferred)${NC}"

INPUT_TRAIN=$(select_file "📥 Select PRE-TRAINING movie (.tif/.tiff)" "Data/")
INPUT_VAL=$(select_file   "🧪 Select VALIDATION movie (.tif/.tiff)" "Data/")
INPUT_MASK=$(select_file  "🎭 Select ANNOTATED MASK (.tif/.tiff)"  "Data/")

chmod +r "$INPUT_TRAIN" "$INPUT_VAL" "$INPUT_MASK" 2>/dev/null || true

echo -e "${YELLOW}Using files:${NC}"
echo "  train: $INPUT_TRAIN"
echo "  valid: $INPUT_VAL"
echo "  mask : $INPUT_MASK"

echo -e "${BLUE}Choose classifier head architecture (event classifier):${NC}"
echo "   1) linear"
echo "   2) minimal"
echo "   3) resnet"
read -rp "Select classifier head (1, 2, or 3) [1]: " CLS_HEAD_CHOICE
case "$CLS_HEAD_CHOICE" in
  2) CLS_HEAD_ARCH="minimal" ;;
  3) CLS_HEAD_ARCH="resnet" ;;
  *) CLS_HEAD_ARCH="linear" ;;
esac
echo -e "${YELLOW}Classifier head: ${CLS_HEAD_ARCH}${NC}"

BACKBONE="unet"

read -rp "$(center_text '📐 Crop size (e.g., 48):')" CROP_SIZE
read -rp "$(center_text '🔬 Pixel resolution (e.g., 0.65):')" PIXEL_RES
read -rp "$(center_text '🔁 Pretraining epochs for 01_fine-tune (--epochs):')" PRETRAIN_EPOCHS
read -rp "$(center_text '🧠 Classifier epochs for 03_event_classification (--training_epochs):')" CLASSIFIER_EPOCHS
read -rp "$(center_text '🧪 Train samples per epoch (e.g., 50000):')" TRAIN_SAMPLES_PER_EPOCH
read -rp "$(center_text '📦 Batch size (e.g., 108):')" BATCHSIZE
read -rp "$(center_text '🖼️ CAM canvas size (cam_size, e.g., 960):')" CAM_SIZE
read -rp "$(center_text '📂 Base output directory (default: outputs) [press Enter]:')" OUTDIR
OUTDIR=${OUTDIR:-outputs}
read -rp "$(center_text '🎲 Random seed:')" SEED
read -rp "$(center_text '🔸 Minimum # pixels in event mask to count as event (min_pixels, e.g., 10):')" MIN_PIXELS
read -rp "$(center_text '🔹 Balanced sample size per class for training (e.g., 50000):')" BALANCED_SAMPLE_SIZE

re_int='^[0-9]+$'
re_float='^[0-9]+(\.[0-9]+)?$'
[ ! -f "$INPUT_TRAIN" ] && echo -e "${RED}❌ Training movie not found at '$INPUT_TRAIN'${NC}" && exit 1
[ ! -f "$INPUT_VAL" ]   && echo -e "${RED}❌ Validation movie not found at '$INPUT_VAL'${NC}" && exit 1
[ ! -f "$INPUT_MASK" ]  && echo -e "${RED}❌ Mask not found at '$INPUT_MASK'${NC}" && exit 1
[[ ! "$CROP_SIZE" =~ $re_int ]] && echo -e "${RED}❌ Crop size must be integer.${NC}" && exit 1
[[ ! "$PRETRAIN_EPOCHS" =~ $re_int ]] && echo -e "${RED}❌ Pretraining epochs must be integer.${NC}" && exit 1
[[ ! "$CLASSIFIER_EPOCHS" =~ $re_int ]] && echo -e "${RED}❌ Classifier epochs must be integer.${NC}" && exit 1
[[ ! "$BATCHSIZE" =~ $re_int ]] && echo -e "${RED}❌ Batch size must be integer.${NC}" && exit 1
[[ ! "$CAM_SIZE" =~ $re_int ]] && echo -e "${RED}❌ cam_size must be integer.${NC}" && exit 1
[[ ! "$TRAIN_SAMPLES_PER_EPOCH" =~ $re_int ]] && echo -e "${RED}❌ train_samples_per_epoch must be integer.${NC}" && exit 1
[[ ! "$SEED" =~ $re_int      ]] && echo -e "${RED}❌ Seed must be integer.${NC}" && exit 1
[[ ! "$PIXEL_RES" =~ $re_float ]] && echo -e "${RED}❌ Pixel resolution must be float.${NC}" && exit 1

mkdir -p "$OUTDIR" || { echo -e "${RED}❌ Cannot create output directory: $OUTDIR${NC}"; exit 1; }

PROJECT_ID="$(basename "$INPUT_TRAIN")"
PROJECT_ID="${PROJECT_ID%.*}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="${TIMESTAMP}_seed${SEED}_cls-${CLS_HEAD_ARCH}"

BASE_DIR="${OUTDIR}/${PROJECT_ID}"
SELECTED_DIR="${BASE_DIR}/${RUN_ID}"

CONFIG_DIR="${SELECTED_DIR}/config"
LOG_DIR="${SELECTED_DIR}/logs"

PH1="${SELECTED_DIR}/phase1_pretraining"
PH2="${SELECTED_DIR}/phase2_data_prep"
PH3="${SELECTED_DIR}/phase3_supervised_classification"
PH4="${SELECTED_DIR}/phase4_mistake_analysis"
PH5="${SELECTED_DIR}/phase5_grad_cam"

MODELS_DIR="${SELECTED_DIR}/models"
SS_MODEL_DIR="${MODELS_DIR}/self_supervised"
CLS_MODEL_DIR="${MODELS_DIR}/supervised"

FIGURES_DIR="${SELECTED_DIR}/figures"
METRICS_DIR="${SELECTED_DIR}/metrics"

mkdir -p \
  "$CONFIG_DIR" "$LOG_DIR" \
  "$PH1" "$PH2" "$PH3" "$PH4" "$PH5" \
  "$MODELS_DIR" "$SS_MODEL_DIR" "$CLS_MODEL_DIR" \
  "$FIGURES_DIR" "$METRICS_DIR" || {
    echo -e "${RED}❌ Cannot create structured output directories under: $SELECTED_DIR${NC}"
    exit 1
  }

MODEL_ID="${PROJECT_ID}_${BACKBONE}_${RUN_ID}"
MODEL_RUN_DIR="${PH1}/model_artifacts"
mkdir -p "$MODEL_RUN_DIR"

echo -e "${GREEN}📁 Run directory:${NC} $SELECTED_DIR"

DEVICE=$(python3 - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)

GPU_ARG_FOR_01=$(python3 - <<'PY'
import torch
print("0" if torch.cuda.is_available() else "cpu")
PY
)

echo "Using device: $DEVICE"

CONFIG_FILE="${CONFIG_DIR}/run_config.yaml"
cat <<EOL > "$CONFIG_FILE"
name: $MODEL_ID
epochs: $PRETRAIN_EPOCHS
augment: 5
batchsize: $BATCHSIZE
size: $CROP_SIZE
cam_size: $CAM_SIZE
backbone: $BACKBONE
features: 32
train_samples_per_epoch: $TRAIN_SAMPLES_PER_EPOCH
num_workers: 4
projhead: minimal_batchnorm
classhead: $CLS_HEAD_ARCH
input_train:
  - "$INPUT_TRAIN"
input_val:
  - "$INPUT_VAL"
input_mask:
  - "$INPUT_MASK"
split_train:
  - [0.0, 1.0]
split_val:
  - [0.0, 1.0]
outdir: "$MODEL_RUN_DIR"
gpu: "$DEVICE"
seed: $SEED
pixel_resolution: $PIXEL_RES
tensorboard: true
write_final_cams: false
binarize: false
min_pixels: $MIN_PIXELS
config_yaml: "$CONFIG_FILE"
EOL

FINETUNE_CONFIG="${CONFIG_DIR}/finetune_config.yaml"
cat <<EOL > "$FINETUNE_CONFIG"
name: $MODEL_ID
input_train:
  - "$INPUT_TRAIN"
input_val:
  - "$INPUT_VAL"
split_train:
  - [0.0, 1.0]
split_val:
  - [0.0, 1.0]
epochs: $PRETRAIN_EPOCHS
seed: $SEED
backbone: $BACKBONE
projhead: minimal_batchnorm
classhead: minimal
perm_equiv: true
features: 32
outdir: "$MODEL_RUN_DIR"
size: $CROP_SIZE
cam_size: $CAM_SIZE
batchsize: $BATCHSIZE
train_samples_per_epoch: $TRAIN_SAMPLES_PER_EPOCH
num_workers: 4
gpu: $GPU_ARG_FOR_01
tensorboard: true
binarize: false
augment: 5
EOL

center_text "${GREEN}📝 Configuration saved to $CONFIG_FILE${NC}"
echo -e "${GREEN}📝 Fine-tune config saved to $FINETUNE_CONFIG${NC}"

DEVICE_FLAG=""
if [[ "$DEVICE" != cuda:* ]]; then
  export CUDA_VISIBLE_DEVICES=""
  DEVICE_FLAG="--device cpu"
fi

SKIP_FINETUNE=""
TAP_MODEL_DIR=""

echo -e "${BLUE}Do you already have a pre-trained TAP model to use? (y/n)${NC}"
read -r HAVE_PRETRAINED
if [[ "$HAVE_PRETRAINED" =~ ^[Yy]$ ]]; then
  echo -e "${YELLOW}Searching for model folders / weights under current project...${NC}"
  PT_DIR=$(select_pretrained_model ".")
  if [ -n "$PT_DIR" ] && [ -d "$PT_DIR" ]; then
    TAP_MODEL_DIR="$PT_DIR"
    SKIP_FINETUNE="1"
    echo -e "${GREEN}✅ Using pre-trained TAP model folder:${NC} $TAP_MODEL_DIR"
  else
    echo -e "${RED}❌ Could not resolve a valid model directory. Proceeding to fine-tune from scratch.${NC}"
  fi
fi

PHASE1_METRICS="${LOG_DIR}/01_finetune_metrics.csv"
echo "step_name,elapsed_sec,peak_ram_mb,disk_after_mb" > "$PHASE1_METRICS"

if [[ -z "$SKIP_FINETUNE" ]]; then
  center_text "${YELLOW}🚀 Phase 1: Fine-tune Model${NC}"
  run_phase "Phase1_Fine-tune" "${LOG_DIR}/01_finetune.log" "$SELECTED_DIR" "$PHASE1_METRICS" \
    python3 Scr/01_fine-tune.py --config "$FINETUNE_CONFIG" --gpu "$GPU_ARG_FOR_01"
fi

if [[ -z "$TAP_MODEL_DIR" ]]; then
  if compgen -G "$MODEL_RUN_DIR/*_backbone_${BACKBONE}" > /dev/null; then
    TAP_MODEL_DIR="$(ls -dt "$MODEL_RUN_DIR"/*_backbone_${BACKBONE} 2>/dev/null | head -1)"
  fi
  if [[ -z "$TAP_MODEL_DIR" ]]; then
    first_pt="$(ls -dt "$MODEL_RUN_DIR"/*.pt 2>/dev/null | head -1 || true)"
    [[ -n "$first_pt" ]] && TAP_MODEL_DIR="$(dirname "$first_pt")"
  fi
fi

echo "TAP model folder (resolved): ${TAP_MODEL_DIR:-<not found>}"
if [[ -n "$TAP_MODEL_DIR" && -d "$TAP_MODEL_DIR" ]]; then
  rsync -a "$TAP_MODEL_DIR"/ "$SS_MODEL_DIR"/ 2>/dev/null || cp -R "$TAP_MODEL_DIR"/. "$SS_MODEL_DIR"/ 2>/dev/null || true
fi

PIPE_METRICS_CSV="${LOG_DIR}/pipeline_metrics.csv"
echo "step_name,elapsed_sec,peak_ram_mb,disk_after_mb" > "$PIPE_METRICS_CSV"

START_TIME=$(date +%s)

FRAMES=2
SUBSAMPLE=1
CROPS_PER_IMAGE="$BATCHSIZE"

center_text "${YELLOW}🚀 Phase 2: Data Preparation${NC}"
run_phase "Phase2_DataPrep" "${LOG_DIR}/02_dataprep.log" "$SELECTED_DIR" "$PIPE_METRICS_CSV" \
  python3 Scr/02_data_prep.py \
    --input_frame "$INPUT_TRAIN" \
    --input_mask "$INPUT_MASK" \
    --cam_size "$CAM_SIZE" \
    --frames "$FRAMES" \
    --subsample "$SUBSAMPLE" \
    --balanced_sample_size "$BALANCED_SAMPLE_SIZE" \
    --crops_per_image "$CROPS_PER_IMAGE" \
    --data_save_dir "$PH2" \
    --size "$CROP_SIZE" \
    --data_seed "$SEED" \
    --pixel_area_threshold "$MIN_PIXELS"

center_text "${YELLOW}🚀 Phase 3: Event Classification${NC}"
run_phase "Phase3_Classification" "${LOG_DIR}/03_classification.log" "$SELECTED_DIR" "$PIPE_METRICS_CSV" \
  python3 Scr/03_event_classification.py \
    --input_frame "$INPUT_VAL" \
    --input_mask "$INPUT_MASK" \
    --cam_size "$CAM_SIZE" \
    --size "$CROP_SIZE" \
    --batchsize "$BATCHSIZE" \
    --training_epochs "$CLASSIFIER_EPOCHS" \
    --balanced_sample_size "$BALANCED_SAMPLE_SIZE" \
    --crops_per_image "$BATCHSIZE" \
    --model_seed "$SEED" \
    --data_seed "$SEED" \
    --dataset_save_dir "$PH2" \
    --num_runs 1 \
    --model_save_dir "$CLS_MODEL_DIR" \
    --model_id "$MODEL_ID" \
    --cls_head_arch "$CLS_HEAD_ARCH" \
    --TAP_model_load_path "$TAP_MODEL_DIR" \
    $DEVICE_FLAG

if [ -d "results/supervised_classification/figures/$MODEL_ID" ]; then
  mkdir -p "$FIGURES_DIR/supervised"
  rsync -a "results/supervised_classification/figures/$MODEL_ID"/ "$FIGURES_DIR/supervised"/ 2>/dev/null || \
  cp -R "results/supervised_classification/figures/$MODEL_ID"/. "$FIGURES_DIR/supervised"/ 2>/dev/null || true
fi

center_text "${YELLOW}🚀 Phase 4: Examining Mistaken Predictions${NC}"
run_phase "Phase4_MistakeAnalysis" "${LOG_DIR}/04_mistake_analysis.log" "$SELECTED_DIR" "$PIPE_METRICS_CSV" \
  python3 Scr/04_examine_mistaken_predictions.py \
    --mistake_pred_dir "$PH4" \
    --masks_path "$INPUT_MASK" \
    --TAP_model_load_path "$TAP_MODEL_DIR" \
    --patch_size "$CROP_SIZE" \
    --test_data_load_path "$PH2/test_data_crops_flat.pth" \
    --combined_model_load_dir "$CLS_MODEL_DIR" \
    --model_id "$MODEL_ID" \
    --cls_head_arch "$CLS_HEAD_ARCH" \
    --num_egs_to_show 10 \
    --save_data

echo -e "${BLUE}Do you want to generate Grad-CAM outputs for ALL frames? (y/n)${NC}"
read -r RUN_GRADCAM
if [[ "$RUN_GRADCAM" =~ ^[Yy]$ ]]; then
  if [[ -z "$TAP_MODEL_DIR" || ! -d "$TAP_MODEL_DIR" ]]; then
    echo -e "${RED}❌ Grad-CAM skipped: trained model directory not found.${NC}"
  else
    center_text "${YELLOW}🟠 Phase 5: Grad-CAM visualizations${NC}"
    GC_BASE_DIR="$PH5"
    GC_VIS_DIR="$GC_BASE_DIR/visuals_full"
    mkdir -p "$GC_VIS_DIR" || { echo -e "${RED}❌ Cannot create $GC_VIS_DIR${NC}"; exit 1; }

    read -r N_FRAMES IMG_H IMG_W <<< "$(
python3 - "$INPUT_VAL" <<'PY'
import sys
from tifffile import TiffFile
path = sys.argv[1]
with TiffFile(path) as tf:
    pages = tf.pages
    n = len(pages)
    p0 = pages[0]
    try:
        h, w = p0.shape[-2], p0.shape[-1]
    except Exception:
        h = getattr(p0, 'imagelength', 1024)
        w = getattr(p0, 'imagewidth', 1024)
print(n, h, w)
PY
)"
    N_FRAMES=${N_FRAMES:-1}
    IMG_H=${IMG_H:-1024}
    IMG_W=${IMG_W:-1024}

    echo -e "${YELLOW}Frames detected:${NC} $N_FRAMES   ${YELLOW}Size:${NC} ${IMG_W}x${IMG_H}"

    GC_DEVICE_FLAG=""
    if [[ "$DEVICE" == mps ]]; then
      GC_DEVICE_FLAG="--device mps"
    elif [[ "$DEVICE" == cuda:* ]]; then
      GC_DEVICE_FLAG="--device $DEVICE"
    else
      export CUDA_VISIBLE_DEVICES=""
      GC_DEVICE_FLAG="--device cpu"
    fi

    run_phase "Phase5_GradCAM_CreateVisuals" "${LOG_DIR}/05_gradcam_create_visuals.log" "$SELECTED_DIR" "$PIPE_METRICS_CSV" \
      env PYTHONPATH=TAP/tarrow python TAP/tarrow/tarrow/visualizations/create_visuals.py \
        --input "$INPUT_VAL" \
        --model "$TAP_MODEL_DIR" \
        --outdir "$GC_VIS_DIR" \
        --delta 1 \
        --frames 2 \
        --size "$IMG_W" "$IMG_H" \
        --n_images "$N_FRAMES" \
        --clip \
        --norm_cam true \
        --file_format tiff \
        $GC_DEVICE_FLAG

    if [ -d "$GC_VIS_DIR/raws" ] && [ -d "$GC_VIS_DIR/cam" ]; then
      GC_FIG_DIR="$GC_BASE_DIR/figure_insets"
      mkdir -p "$GC_FIG_DIR"

      run_phase "Phase5_GradCAM_Insets" "${LOG_DIR}/06_gradcam_insets.log" "$SELECTED_DIR" "$PIPE_METRICS_CSV" \
        bash -lc '
set -e
shopt -s nullglob
RAW_DIR="'"$GC_VIS_DIR"'/raws"
CAM_DIR="'"$GC_VIS_DIR"'/cam"
OUT_DIR="'"$GC_FIG_DIR"'"
COUNT=0
for RAW in "$RAW_DIR"/raws_*.tif "$RAW_DIR"/raws_*.tiff; do
  [ -f "$RAW" ] || continue
  base=$(basename "$RAW")
  idx=${base#raws_}
  CAM="$CAM_DIR/cam_$idx"
  if [ ! -f "$CAM" ]; then
    if [[ "$CAM" == *.tiff ]]; then
      alt="${CAM%.tiff}.tif"
    else
      alt="${CAM%.tif}.tiff"
    fi
    [ -f "$alt" ] && CAM="$alt"
  fi
  [ -f "$CAM" ] || { echo "Skipping $idx (no CAM)"; continue; }
  PYTHONPATH=TAP/tarrow python TAP/tarrow/tarrow/visualizations/cam_insets.py \
    --dataset custom \
    --img_raw "$RAW" \
    --img_cam "$CAM" \
    --outdir "$OUT_DIR" \
    --n_insets 8 \
    --width_insets 96 \
    --frames 2 \
    --delta 1 \
    --horiz || true
  COUNT=$((COUNT+1))
done
echo "✅ Inset figures generated for $COUNT frames into: $OUT_DIR"
'
      echo -e "${GREEN}✅ Grad-CAM outputs saved under: ${NC}$GC_BASE_DIR"
    else
      echo -e "${RED}❌ Expected Grad-CAM folders missing under $GC_VIS_DIR (raws/ and cam/).${NC}"
    fi
  fi
else
  echo -e "${YELLOW}Skipping Grad-CAM generation by user choice.${NC}"
fi

cp "$PIPE_METRICS_CSV" "$METRICS_DIR/pipeline_metrics.csv" 2>/dev/null || true
cp "$PHASE1_METRICS" "$METRICS_DIR/finetune_metrics.csv" 2>/dev/null || true

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

center_text "${GREEN}🎉 EDITS Pipeline Complete!${NC}"
echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"
echo "🔹 Project ID          : $PROJECT_ID"
echo "🔹 Run ID              : $RUN_ID"
echo "🔹 Model ID            : $MODEL_ID"
echo "🔹 Run Root Dir        : $SELECTED_DIR"
echo "🔹 Config Dir          : $CONFIG_DIR"
echo "🔹 Logs Dir            : $LOG_DIR"
echo "🔹 Phase1 (Pretrain)   : $PH1"
echo "🔹 Phase2 (Data Prep)  : $PH2"
echo "🔹 Phase3 (Classif)    : $PH3"
echo "🔹 Phase4 (Mistakes)   : $PH4"
echo "🔹 Phase5 (Grad-CAM)   : $PH5"
echo "🔹 Models (self-sup)   : $SS_MODEL_DIR"
echo "🔹 Models (supervised) : $CLS_MODEL_DIR"
echo "🔹 Figures             : $FIGURES_DIR"
echo "🔹 Metrics             : $METRICS_DIR"
echo "🔹 TAP Model Dir (raw) : ${TAP_MODEL_DIR:-<not found>}"
echo "🔹 Crop Size           : $CROP_SIZE"
echo "🔹 Pretrain Epochs (01): $PRETRAIN_EPOCHS"
echo "🔹 Classifier Epochs(03): $CLASSIFIER_EPOCHS"
echo "🔹 Pixel Res           : $PIXEL_RES"
echo "🔹 Backbone            : $BACKBONE"
echo "🔹 Batch size          : $BATCHSIZE"
echo "🔹 cam_size            : $CAM_SIZE"
echo "🔹 train_samples/epoch : $TRAIN_SAMPLES_PER_EPOCH"
echo "🔹 Mask File           : $INPUT_MASK"
echo "🔹 Config File         : $CONFIG_FILE"
echo "🔹 Pipeline Metrics    : $PIPE_METRICS_CSV"
echo "⏱️  Runtime            : $((RUNTIME / 60)) min $((RUNTIME % 60)) sec"
echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"

center_text "${YELLOW}📝 Generating HTML Report${NC}"
python3 Scr/05_generate_report.py --config "$CONFIG_FILE" --outdir "$SELECTED_DIR"

REPORT_PATH="$SELECTED_DIR/report.html"
echo -e "${GREEN}Attempting to open your report in your browser...${NC}"
open_html_report "$REPORT_PATH"

echo -e "${BLUE}─────────────────────────────────────────────${NC}"
echo -e "${GREEN}🎯 EDITS FINAL RESULTS SUMMARY${NC}"
echo -e "${BLUE}─────────────────────────────────────────────${NC}"
echo -e "${YELLOW}🔸 Run root directory :${NC} $SELECTED_DIR"
echo -e "${YELLOW}🔸 HTML report         :${NC} $REPORT_PATH"
echo -e "${YELLOW}🔸 Configs             :${NC} $CONFIG_DIR"
echo -e "${YELLOW}🔸 Logs                :${NC} $LOG_DIR"
echo -e "${YELLOW}🔸 Phase1 (pretrain)   :${NC} $PH1"
echo -e "${YELLOW}🔸 Phase2 (data prep)  :${NC} $PH2"
echo -e "${YELLOW}🔸 Phase3 (classif)    :${NC} $PH3"
echo -e "${YELLOW}🔸 Phase4 (mistakes)   :${NC} $PH4"
echo -e "${YELLOW}🔸 Phase5 (grad_cam)   :${NC} $PH5"
echo -e "${YELLOW}🔸 Models (self-sup)   :${NC} $SS_MODEL_DIR"
echo -e "${YELLOW}🔸 Models (supervised) :${NC} $CLS_MODEL_DIR"
echo -e "${YELLOW}🔸 Figures             :${NC} $FIGURES_DIR"
echo -e "${YELLOW}🔸 Metrics             :${NC} $METRICS_DIR"
echo -e "${BLUE}─────────────────────────────────────────────${NC}"
echo -e "${GREEN}Open your report in your browser:${NC} file://$REPORT_PATH"
