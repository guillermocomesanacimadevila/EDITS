#!/bin/bash
# run_tap_conda.sh

# Update: Allows classifier head choice (linear, minimal, resnet)
# Update: Sectioned, colorized, metrics helpers, future grid search extensible.

# ──────────────────────────────────────────────────────────────── #
#                 CellFate PIPELINE: SELF-CONFIGURING              #
#      (Conda auto-install + env bootstrap + pipeline)             #
# ──────────────────────────────────────────────────────────────── #



# ───────────── Terminal Colors ───────────── #
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# ───────────── Detect WSL ───────────── #
IS_WSL=false
if grep -qi microsoft /proc/version 2>/dev/null; then
    IS_WSL=true
fi

# ───────────── Bootstrap System Requirements (WSL/Ubuntu/Mac) ───────────── #
echo -e "${BLUE}🔧 Checking and installing system requirements...${NC}"

SYSTEM_TOOLS=(python3 pip wget sudo git du)
APT_PACKAGES=(python3 python3-pip wget sudo git du time fontconfig fonts-dejavu-core fzf)
NEED_TO_INSTALL=()

for cmd in "${SYSTEM_TOOLS[@]}"; do
    command -v "$cmd" >/dev/null 2>&1 || NEED_TO_INSTALL+=("$cmd")
done

# Only do apt-get for Linux/WSL
OS_TYPE="$(uname)"
if [[ "$OS_TYPE" == "Linux" ]] && command -v apt-get &>/dev/null; then
    echo -e "${BLUE}➡️  Ensuring essential apt packages: ${YELLOW}${APT_PACKAGES[*]}${NC}"
    sudo apt-get update -qq
    sudo apt-get install -y "${APT_PACKAGES[@]}"
    # WSL: ensure wslu/wslview for browser open
    if $IS_WSL && ! command -v wslview &>/dev/null; then
        sudo apt-get install -y wslu
    fi
fi

# Mac users: Homebrew block
if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo -e "${BLUE}➡️  On Mac? Ensure Homebrew is installed: https://brew.sh"
    for brewdep in python3 wget git fzf fontconfig; do
        brew list "$brewdep" >/dev/null 2>&1 || brew install "$brewdep"
    done
fi

# ───────────── HTML Report Opener (WSL+Linux+Mac) ───────────── #
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
      echo -e "${YELLOW}⚠️  Could not auto-open HTML report. Please open manually:${NC} file://$html_path"
    fi
  else
    echo -e "${RED}❌ HTML report not found: $html_path${NC}"
  fi
}

# ───────────── Help Option ───────────── #
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo -e "${YELLOW}CellFate PIPELINE${NC}"
    echo -e "Usage: bash $0"
    echo "You will be interactively prompted for input files/parameters."
    echo "Outputs go to ./runs/ and to your specified output directory."
    echo -e "After run, open your HTML report in your browser.\n"
    exit 0
fi

# ───────────── Direct Grid Search Mode ───────────── #
if [[ "$1" == "--grid-only" ]]; then
    echo -e "${BLUE}Running full spatial hyperparameter grid search...${NC}"
    python3 Workflow/grid_search_spatial.py
    exit 0
fi

trap 'echo -e "\n${RED}⚡️ Script interrupted by user. Exiting!${NC}"; exit 1' SIGINT

# ───────────── Conda/Miniconda Bootstrap ───────────── #
ENV_NAME="cellfate-env"
ENV_YML="environment.yml"

if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}🔄 Conda not found. Installing Miniconda...${NC}"
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    export PATH="$HOME/miniconda/bin:$PATH"
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    if ! command -v conda &> /dev/null; then
      echo -e "${RED}❌ Miniconda installation failed or PATH not updated.${NC}"
      exit 1
    fi
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
    conda env create -f "$ENV_YML" -n "$ENV_NAME"
    if [ $? -ne 0 ]; then
      echo -e "${RED}❌ Failed to create conda environment '$ENV_NAME'.${NC}"
      exit 1
    fi
    echo -e "${GREEN}✅ Conda environment '$ENV_NAME' created.${NC}"
fi

echo -e "${GREEN}🔄 Activating '$ENV_NAME'...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo -e "${RED}❌ Failed to activate '$ENV_NAME'!${NC}"; exit 1; }

# ───────────── FONT FIX FOR MATPLOTLIB ───────────── #
mkdir -p ~/.config/matplotlib
echo "font.family: sans-serif
font.sans-serif: DejaVu Sans
" > ~/.config/matplotlib/matplotlibrc

mkdir -p ~/.cache/matplotlib
chmod 700 ~/.cache/matplotlib

# --- Preflight check for critical Python packages (fail early if broken env) ---
python3 -c "import torch, numpy, matplotlib" 2>/dev/null || {
  echo -e "${RED}❌ Required Python packages missing (torch, numpy, matplotlib). Check your Conda env!${NC}"
  exit 1
}

# ───────────── TAP/tarrow install (editable mode) ───────────── #
echo -e "${YELLOW}🔗 Installing TAP/tarrow package in editable mode (if not already)...${NC}"
if [ -d "TAP/tarrow" ] && [ -f "TAP/tarrow/setup.py" ]; then
    pip show tarrow > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        pip install -e TAP/tarrow || { echo -e "${RED}❌ Failed to install TAP/tarrow.${NC}"; exit 1; }
    fi
else
    echo -e "${RED}❌ TAP/tarrow directory or setup.py not found!${NC}"
    exit 1
fi

# ───────────── Version Logging ───────────── #
echo -e "${YELLOW}🔢 Environment Versions:${NC}"
echo -n "Python: "; python3 --version
echo -n "Conda: "; conda --version
echo -n "PyTorch: "; python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "N/A"
echo -n "Numpy: "; python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo "N/A"
echo "----------------------------"

# ───────────── Check Write Permission to Home and Runs Directory ───────────── #
if [ ! -w "$HOME" ]; then
  echo -e "${RED}❌ No write permission to home directory: $HOME${NC}"
  exit 1
fi

if [ ! -d "runs" ]; then
  mkdir -p runs || { echo -e "${RED}❌ Cannot create 'runs' directory.${NC}"; exit 1; }
elif [ ! -w "runs" ]; then
  echo -e "${RED}❌ No write permission to 'runs' directory.${NC}"
  exit 1
fi

# ───────────── Check Write Permission to Home and Runs Directory ───────────── #
if [ ! -w "$HOME" ]; then
  echo -e "${RED}❌ No write permission to home directory: $HOME${NC}"
  exit 1
fi

if [ ! -d "runs" ]; then
  mkdir -p runs || { echo -e "${RED}❌ Cannot create 'runs' directory.${NC}"; exit 1; }
elif [ ! -w "runs" ]; then
  echo -e "${RED}❌ No write permission to 'runs' directory.${NC}"
  exit 1
fi


# ───────────── fzf Auto-Installer (Linux/macOS) ───────────── #
if ! command -v fzf &> /dev/null; then
    echo -e "${YELLOW}⚙️  fzf not found. Installing it for better file selection...${NC}"
    OS_TYPE="$(uname)"
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        if command -v brew &> /dev/null; then
            echo -e "${BLUE}➡️  Using Homebrew to install fzf (macOS)...${NC}"
            brew install fzf || { echo -e "${RED}❌ Failed to install fzf via brew.${NC}"; exit 1; }
        else
            echo -e "${RED}❌ Homebrew not found. Install it from https://brew.sh first.${NC}"
            exit 1
        fi
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        if command -v apt-get &> /dev/null; then
            echo -e "${BLUE}➡️  Using apt-get to install fzf (Linux)...${NC}"
            sudo apt-get update -qq
            sudo apt-get install -y fzf || { echo -e "${RED}❌ Failed to install fzf via apt.${NC}"; exit 1; }
        else
            echo -e "${RED}❌ apt-get not available. Install fzf manually.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Unsupported OS: $OS_TYPE. Install fzf manually.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ fzf installed successfully.${NC}"
fi

# ──────────────────────────────────────────────────────────────── #
#         USER-FRIENDLY FILE SELECTION (fzf or fallback)          #
# ──────────────────────────────────────────────────────────────── #
select_file() {
    local prompt="$1"
    local start_dir="$2"
    local file
    if command -v fzf &> /dev/null; then
        file=$(find "$start_dir" -type f -name "*.tif" | fzf --prompt="$prompt " --height=15 --border)
        if [ -z "$file" ]; then
            echo -e "${RED}❌ No file selected! Exiting.${NC}"
            exit 1
        fi
        if [[ "$file" == "$PWD"* ]]; then
            file="."${file#$PWD}
        fi
        echo "$file"
    else
        echo -e "${YELLOW}Tip: Install 'fzf' for interactive file picking (brew install fzf / apt-get install fzf).${NC}"
        echo -e "${YELLOW}Available files in $start_dir:${NC}"
        find "$start_dir" -type f -name "*.tif" | nl
        read -p "$prompt (copy-paste or type relative path): " file
        if [ ! -f "$file" ]; then
            echo -e "${RED}❌ File not found: $file${NC}"
            exit 1
        fi
        echo "$file"
    fi
}

select_dir() {
    local prompt="$1"
    local start_dir="$2"
    local dir
    if command -v fzf &> /dev/null; then
        dir=$(find "$start_dir" -type d | fzf --prompt="$prompt " --height=15 --border)
        if [ -z "$dir" ]; then
            echo -e "${RED}❌ No directory selected! Exiting.${NC}"
            exit 1
        fi
        if [[ "$dir" == "$PWD"* ]]; then
            dir="."${dir#$PWD}
        fi
        echo "$dir"
    else
        echo -e "${YELLOW}Available directories in $start_dir:${NC}"
        find "$start_dir" -type d | nl
        read -p "$prompt (copy-paste or type relative path): " dir
        if [ ! -d "$dir" ]; then
            echo -e "${RED}❌ Directory not found: $dir${NC}"
            exit 1
        fi
        echo "$dir"
    fi
}

# ──────────────────────────────────────────────────────────────── #
#        METRICS LOGGING HELPERS (CROSS-PLATFORM TIMER)           #
# ──────────────────────────────────────────────────────────────── #
run_and_log() {
  local STEP="$1"
  local LOG="$2"
  local DISK_DIR="$3"
  local CSV_FILE="$4"
  shift 4
  local CMD=("$@")
  local START END ELAPSED_SEC
  local PEAK_RAM_MB="NA"
  if command -v /usr/bin/time &> /dev/null; then
    /usr/bin/time -v "${CMD[@]}" 2> "$LOG"
    local ELAPSED=$(grep "Elapsed (wall clock) time" "$LOG" | awk '{print $8}')
    local h=0 m=0 s=0
    if [[ "$ELAPSED" == *:*:* ]]; then
        IFS=: read -r h m s <<< "$ELAPSED"
    elif [[ "$ELAPSED" == *:* ]]; then
        IFS=: read -r m s <<< "$ELAPSED"
        h=0
    elif [[ -n "$ELAPSED" ]]; then
        s="$ELAPSED"
        h=0
        m=0
    fi
    h=${h:-0}; m=${m:-0}; s=${s:-0}
    h=$(echo "$h" | sed 's/^0*//'); h=${h:-0}
    m=$(echo "$m" | sed 's/^0*//'); m=${m:-0}
    s=${s%%.*}
    ELAPSED_SEC=$((10#$h*3600 + 10#$m*60 + 10#$s))
    PEAK_RAM_MB=$(grep "Maximum resident set size" "$LOG" | awk '{print int($6/1024)}')
  else
    echo -e "${YELLOW}⚠️  /usr/bin/time not found! Falling back to builtin 'time'. Resource usage will be limited.${NC}"
    START=$(date +%s)
    "${CMD[@]}"
    END=$(date +%s)
    ELAPSED_SEC=$((END - START))
  fi
  local DISK_MB=$(du -sm "$DISK_DIR" 2>/dev/null | awk '{print $1}')
  echo "$STEP,$ELAPSED_SEC,$PEAK_RAM_MB,$DISK_MB" >> "$CSV_FILE"
}

center_text() {
    local width=70
    local text="$1"
    printf "\n%*s\n\n" $(( (${#text} + width) / 2 )) "$text"
}

# ──────────────────────────────────────────────────────────────── #
#                   MAIN USER INTERACTION SECTION                 #
# ──────────────────────────────────────────────────────────────── #

center_text "${BLUE}🔬 CellFate ML Pipeline Setup${NC}"
echo -e "${YELLOW}ℹ️  Select files/directories interactively below (relative paths preferred)${NC}"

# ──────── SPATIAL GRID SEARCH OPTION (new section) ──────── #
echo -e "${BLUE}Do you want to run the full spatial hyperparameter grid search? (y/n)${NC}"
read -r RUN_SPATIAL_GRID
if [[ "$RUN_SPATIAL_GRID" == "y" || "$RUN_SPATIAL_GRID" == "Y" ]]; then
    center_text "${YELLOW}🚀 Running spatial hyperparameter grid search (this will take time)...${NC}"
    python3 Workflow/grid_search_spatial.py
    center_text "${GREEN}✅ Spatial grid search complete!${NC}"
    echo -e "${YELLOW}📄 Results: spatial_search_results/spatial_grid_results.csv${NC}"
    echo -e "${YELLOW}📊 Plots:   spatial_search_results/plots/${NC}"
    exit 0
fi

INPUT_TRAIN=$(select_file "📥 Select PRE-TRAINING movie (.tif)" "Data/")

center_text "${BLUE}🧪 Validate on just 1 movie or on a whole directory?${NC}"
echo "   0: Single validation movie"
echo "   1: Validate on every .tif in a folder (no classifier, TAP metrics only!)"
read -p "Select option (0 or 1): " VAL_BATCH

if [[ "$VAL_BATCH" != "0" && "$VAL_BATCH" != "1" ]]; then
    echo -e "${RED}❌ Invalid input. Please enter 0 or 1.${NC}"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$VAL_BATCH" == "0" ]; then

    # -------------- BALANCING METHOD PROMPT --------------
    echo -e "${BLUE}Choose event balancing method for event classifier:${NC}"
    echo "   1) Standard balanced (random sampling for equal pos/neg)"
    echo "   2) Stratified split + oversampling + augmentation"
    read -p "Select balancing method (1 or 2) [1]: " BAL_CHOICE
    if [[ "$BAL_CHOICE" == "2" ]]; then
      BALANCING_METHOD="stratified_oversample"
    else
      BALANCING_METHOD="balanced"
    fi

    # -------------- CLASSIFIER HEAD ARCHITECTURE PROMPT --------------
    echo -e "${BLUE}Choose classifier head architecture (event classifier):${NC}"
    echo "   1) linear"
    echo "   2) minimal"
    echo "   3) resnet"
    read -p "Select classifier head (1, 2, or 3) [1]: " CLS_HEAD_CHOICE
    case "$CLS_HEAD_CHOICE" in
      2) CLS_HEAD_ARCH="minimal" ;;
      3) CLS_HEAD_ARCH="resnet" ;;
      *) CLS_HEAD_ARCH="linear" ;;
    esac
    # -------------- END ADDITION --------------------------

    INPUT_VAL=$(select_file "🧪 Select VALIDATION movie (.tif)" "Data/")
    INPUT_MASK=$(select_file "🎭 Select ANNOTATED MASK (.tif)" "Data/")
    read -p "$(center_text '📐 Crop size (e.g., 48):')" CROP_SIZE
    read -p "$(center_text '🔬 Pixel resolution (e.g., 0.65):')" PIXEL_RES
    read -p "$(center_text '🔁 Number of training epochs:')" EPOCHS
    read -p "$(center_text '🔂 Number of independent runs (fine-tune only, e.g., 1 or 5):')" NUM_RUNS
    read -p "$(center_text '📂 Output directory path (e.g., Data/toy_data):')" OUTDIR
    read -p "$(center_text '🎲 Random seed:')" SEED
    read -p "$(center_text '🧠 Backbone (unet, spectformer-xs):')" BACKBONE
    read -p "$(center_text '🔸 Minimum # pixels in event mask to count as event (min_pixels, e.g., 10):')" MIN_PIXELS
    read -p "$(center_text '🔹 Balanced sample size per class for training (e.g., 50000):')" BALANCED_SAMPLE_SIZE

    re_int='^[0-9]+$'
    re_float='^[0-9]+(\.[0-9]+)?$'
    [ ! -f "$INPUT_TRAIN" ] && echo -e "${RED}❌ Training movie not found at '$INPUT_TRAIN'${NC}" && exit 1
    [ ! -f "$INPUT_VAL" ] && echo -e "${RED}❌ Validation movie not found at '$INPUT_VAL'${NC}" && exit 1
    [ ! -f "$INPUT_MASK" ] && echo -e "${RED}❌ Mask not found at '$INPUT_MASK'${NC}" && exit 1
    [[ ! "$CROP_SIZE" =~ $re_int ]] && echo -e "${RED}❌ Crop size must be integer.${NC}" && exit 1
    [[ ! "$EPOCHS" =~ $re_int ]] && echo -e "${RED}❌ Epochs must be integer.${NC}" && exit 1
    [[ ! "$NUM_RUNS" =~ $re_int ]] && echo -e "${RED}❌ Number of independent runs must be integer.${NC}" && exit 1
    [[ ! "$SEED" =~ $re_int ]] && echo -e "${RED}❌ Seed must be integer.${NC}" && exit 1
    [[ ! "$PIXEL_RES" =~ $re_float ]] && echo -e "${RED}❌ Pixel resolution must be float.${NC}" && exit 1
    [[ "$BACKBONE" != "unet" && "$BACKBONE" != "spectformer-xs" ]] && echo -e "${RED}❌ Unsupported backbone: $BACKBONE${NC}" && exit 1
    [[ ! "$MIN_PIXELS" =~ $re_int ]] && echo -e "${RED}❌ min_pixels must be integer.${NC}" && exit 1

    outdir_parent=$(dirname "$OUTDIR")
    if [ ! -w "$outdir_parent" ]; then
      echo -e "${RED}❌ No write permission in output directory parent: $outdir_parent${NC}"
      exit 1
    fi

    FT_RUNS=()
    for RUN_IDX in $(seq 1 $NUM_RUNS); do
        RUN_SEED=$SEED
        RUN_ID="${TIMESTAMP}_run${RUN_IDX}"
        MODEL_ID="$(basename "$INPUT_TRAIN" .tif)_${BACKBONE}_${RUN_ID}"
        MODEL_RUN_DIR="runs/${MODEL_ID}"
        CURR_OUTDIR="${OUTDIR%/}_${MODEL_ID}"
        mkdir -p "$MODEL_RUN_DIR" || { echo -e "${RED}❌ Cannot create model run directory: $MODEL_RUN_DIR${NC}"; exit 1; }
        mkdir -p "$CURR_OUTDIR" || { echo -e "${RED}❌ Cannot create output directory: $CURR_OUTDIR${NC}"; exit 1; }
        mkdir -p "$CURR_OUTDIR/figures"

        CONFIG_FILE="$CURR_OUTDIR/run_config.yaml"
        cat <<EOL > "$CONFIG_FILE"
name: $MODEL_ID
epochs: $EPOCHS
augment: 5
batchsize: 108
size: $CROP_SIZE
cam_size: 960
backbone: $BACKBONE
features: 32
train_samples_per_epoch: 50000
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
gpu: "0"
seed: $RUN_SEED
pixel_resolution: $PIXEL_RES
tensorboard: true
write_final_cams: false
binarize: false
min_pixels: $MIN_PIXELS
config_yaml: "$CONFIG_FILE"
EOL

        center_text "${GREEN}📝 Configuration saved to $CONFIG_FILE${NC}"

        LOGFILE="$CURR_OUTDIR/finetune_log.txt"
        METRICS_CSV="$CURR_OUTDIR/finetune_metrics.csv"
        echo "step_name,elapsed_sec,peak_ram_mb,disk_after_mb" > "$METRICS_CSV"
        exec > >(tee -i "$LOGFILE")
        exec 2>&1

        STEP_LOG="$CURR_OUTDIR/01_finetune_timing.log"
        center_text "${YELLOW}🚀 Fine-tune Model (Run $RUN_IDX of $NUM_RUNS)${NC}"
        run_and_log "Fine-tune" "$STEP_LOG" "$CURR_OUTDIR" "$METRICS_CSV" python3 Workflow/01_fine-tune.py --config "$CONFIG_FILE"

        FT_RUNS+=("$CURR_OUTDIR")
        center_text "${GREEN}✔️ Fine-tune #$RUN_IDX complete! Results in $CURR_OUTDIR${NC}"
        echo -e "${YELLOW}---------------------------------------------${NC}"
    done

    # Prompt user to pick which fine-tune result to use for the rest of the pipeline
    echo -e "${BLUE}\nAvailable fine-tuned runs:${NC}"
    select SELECTED_DIR in "${FT_RUNS[@]}"; do
        if [[ -n "$SELECTED_DIR" && -d "$SELECTED_DIR" ]]; then
            break
        fi
        echo -e "${RED}Invalid selection. Choose a number from the list above.${NC}"
    done

    CONFIG_FILE="$SELECTED_DIR/run_config.yaml"
    MODEL_RUN_DIR=$(grep '^outdir:' "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
    MODEL_ID=$(grep '^name:' "$CONFIG_FILE" | awk '{print $2}')
    TAP_MODEL_DIR="${MODEL_RUN_DIR}/${MODEL_ID}_backbone_${BACKBONE}"
    METRICS_CSV="$SELECTED_DIR/pipeline_metrics.csv"
    LOGFILE="$SELECTED_DIR/pipeline_log.txt"
    INPUT_MASK="$INPUT_MASK"
    INPUT_VAL="$INPUT_VAL"
    CROP_SIZE="$CROP_SIZE"

    exec > >(tee -i "$LOGFILE")
    exec 2>&1
    echo "step_name,elapsed_sec,peak_ram_mb,disk_after_mb" > "$METRICS_CSV"
    START_TIME=$(date +%s)

    STEP_LOG="$SELECTED_DIR/02_dataprep_timing.log"
    center_text "${YELLOW}🚀 Data Preparation${NC}"
    run_and_log "Data Preparation" "$STEP_LOG" "$SELECTED_DIR" "$METRICS_CSV" python3 Workflow/02_data_prep.py \
        --input_frame "$INPUT_VAL" \
        --input_mask "$INPUT_MASK" \
        --data_save_dir "$SELECTED_DIR" \
        --size "$CROP_SIZE" \
        --binarize \
        --min_pixels "$MIN_PIXELS" \
        --data_seed "$SEED"

    echo "TAP model folder: $TAP_MODEL_DIR"
    ls -lh "$TAP_MODEL_DIR"

    STEP_LOG="$SELECTED_DIR/03_classification_timing.log"
    center_text "${YELLOW}🚀 Event Classification${NC}"

    # 🟩 Prompt for grid search vs regular run
    echo -e "${BLUE}Do you want to run a hyperparameter grid search for the event classifier? (y/n)${NC}"
    read -r GRID_SEARCH
    if [[ "$GRID_SEARCH" == "y" || "$GRID_SEARCH" == "Y" ]]; then
        # 🟩 Run built-in grid search mode from your Python script
        run_and_log "Event Classification GridSearch" "$STEP_LOG" "$SELECTED_DIR" "$METRICS_CSV" \
            python3 Workflow/03_event_classification.py grid
        echo -e "${GREEN}Grid search complete. Results saved to grid_search_results.csv in working directory.${NC}"
        # Optionally display best result summary
        if [ -f grid_search_results.csv ]; then
            echo -e "${BLUE}Top results from grid_search_results.csv:${NC}"
            head -5 grid_search_results.csv
        fi
    else
        # Standard single run as before
        run_and_log "Event Classification" "$STEP_LOG" "$SELECTED_DIR" "$METRICS_CSV" \
            python3 Workflow/03_event_classification.py \
            --input_frame "$INPUT_VAL" \
            --input_mask "$INPUT_MASK" \
            --cam_size 960 \
            --size "$CROP_SIZE" \
            --batchsize 108 \
            --training_epochs "$EPOCHS" \
            --balanced_sample_size "$BALANCED_SAMPLE_SIZE" \
            --crops_per_image 108 \
            --model_seed "$SEED" \
            --data_seed "$SEED" \
            --data_save_dir "$SELECTED_DIR" \
            --num_runs 1 \
            --model_save_dir "$MODEL_RUN_DIR" \
            --model_id "$MODEL_ID" \
            --cls_head_arch "$CLS_HEAD_ARCH" \
            --backbone "$BACKBONE" \
            --name "$MODEL_ID" \
            --binarize false \
            --TAP_model_load_path "$TAP_MODEL_DIR" \
            --balancing_method "$BALANCING_METHOD"
    fi

    STEP_LOG="$SELECTED_DIR/04_mistake_analysis_timing.log"
    center_text "${YELLOW}🚀 Examining Mistaken Predictions${NC}"
    run_and_log "Mistake Analysis" "$STEP_LOG" "$SELECTED_DIR" "$METRICS_CSV" python3 Workflow/04_examine_mistaken_predictions.py \
        --mistake_pred_dir "$MODEL_RUN_DIR" \
        --masks_path "$INPUT_MASK" \
        --TAP_model_load_path "$TAP_MODEL_DIR" \
        --patch_size "$CROP_SIZE" \
        --test_data_load_path "$SELECTED_DIR/test_data_crops_flat.pth" \
        --combined_model_load_dir "$MODEL_RUN_DIR" \
        --model_id "$MODEL_ID" \
        --cls_head_arch "$CLS_HEAD_ARCH" \
        --num_egs_to_show 10 \
        --save_data

    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    center_text "${GREEN}🎉 CellFate Pipeline Complete for Selected Run!${NC}"
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"
    echo "🔹 Model ID      : $MODEL_ID"
    echo "🔹 Model Dir     : $MODEL_RUN_DIR"
    echo "🔹 Output Dir    : $SELECTED_DIR"
    echo "🔹 Crop Size     : $CROP_SIZE"
    echo "🔹 Epochs        : $EPOCHS"
    echo "🔹 Pixel Res     : $PIXEL_RES"
    echo "🔹 Backbone      : $BACKBONE"
    echo "🔹 Mask File     : $INPUT_MASK"
    echo "🔹 Config File   : $CONFIG_FILE"
    echo "🔹 Log File      : $LOGFILE"
    echo "🔹 Metrics CSV   : $METRICS_CSV"
    echo "⏱️  Runtime: $((RUNTIME / 60)) min $((RUNTIME % 60)) sec"
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"

    center_text "${YELLOW}📝 Generating HTML Report${NC}"
    python3 Workflow/05_generate_report.py --config "$CONFIG_FILE" --outdir "$SELECTED_DIR"

    # ---- Auto-Open HTML report ----
    echo -e "${GREEN}Attempting to open your report in your browser...${NC}"
    open_html_report "$SELECTED_DIR/report.html"

    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${GREEN}🎯 CellFate FINAL RESULTS SUMMARY${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}🔸 Output directory    :${NC} $SELECTED_DIR"
    echo -e "${YELLOW}🔸 Figures directory   :${NC} $SELECTED_DIR/figures"
    echo -e "${YELLOW}🔸 Log file           :${NC} $LOGFILE"
    echo -e "${YELLOW}🔸 Config file        :${NC} $CONFIG_FILE"
    echo -e "${YELLOW}🔸 HTML report        :${NC} $SELECTED_DIR/report.html"
    echo -e "${YELLOW}🔸 Metrics CSV        :${NC} $METRICS_CSV"
    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${GREEN}Open your report in your browser:${NC} file://$SELECTED_DIR/report.html"

    # ======= AUTOMATIC SHADOW SUMMARY PLOT (FULLY INTEGRATED) =======
    METRICS_LIST=()
    for RUN_DIR in "${FT_RUNS[@]}"; do
      MET_CSV="$RUN_DIR/metrics.csv"
      if [ -f "$MET_CSV" ]; then
        if head -1 "$MET_CSV" | grep -q "epoch"; then
          METRICS_LIST+=("$MET_CSV")
        fi
      fi
    done

    if [ "${#METRICS_LIST[@]}" -gt 1 ]; then
        SUMMARY_OUTDIR="${OUTDIR%/}_summary_shadow"
        echo -e "${YELLOW}Generating shadow plots and metrics summary in $SUMMARY_OUTDIR ...${NC}"
        python3 Workflow/01_fine-tune.py --metrics_csv_list "${METRICS_LIST[@]}" --outdir "$SUMMARY_OUTDIR"
        echo -e "${GREEN}Summary shadow plots and mean/STD results saved in $SUMMARY_OUTDIR${NC}"
    fi

else

    # ──────────────────────────────────────────────────────────────── #
    #               BATCH VALIDATION (TAP ONLY) MODE                  #
    # ──────────────────────────────────────────────────────────────── #

    VAL_DIR=$(select_dir "📂 Select directory with validation .tif files" "Data/")
    read -p "$(center_text '📐 Crop size (e.g., 48):')" CROP_SIZE
    read -p "$(center_text '🔬 Pixel resolution (e.g., 0.65):')" PIXEL_RES
    read -p "$(center_text '📂 Output directory path (e.g., Data/toy_data):')" OUTDIR
    read -p "$(center_text '🎲 Random seed:')" SEED
    read -p "$(center_text '🧠 Backbone (unet, spectformer-xs):')" BACKBONE
    read -p "$(center_text '🔸 Minimum # pixels in event mask to count as event (min_pixels, e.g., 10):')" MIN_PIXELS

    re_int='^[0-9]+$'
    re_float='^[0-9]+(\.[0-9]+)?$'
    [ ! -f "$INPUT_TRAIN" ] && echo -e "${RED}❌ Training movie not found at '$INPUT_TRAIN'${NC}" && exit 1
    [ ! -d "$VAL_DIR" ] && echo -e "${RED}❌ Validation directory not found at '$VAL_DIR'${NC}" && exit 1
    [[ ! "$CROP_SIZE" =~ $re_int ]] && echo -e "${RED}❌ Crop size must be integer.${NC}" && exit 1
    [[ ! "$SEED" =~ $re_int ]] && echo -e "${RED}❌ Seed must be integer.${NC}" && exit 1
    [[ ! "$PIXEL_RES" =~ $re_float ]] && echo -e "${RED}❌ Pixel resolution must be float.${NC}" && exit 1
    [[ "$BACKBONE" != "unet" && "$BACKBONE" != "spectformer-xs" ]] && echo -e "${RED}❌ Unsupported backbone: $BACKBONE${NC}" && exit 1
    [[ ! "$MIN_PIXELS" =~ $re_int ]] && echo -e "${RED}❌ min_pixels must be integer.${NC}" && exit 1

    outdir_parent=$(dirname "$OUTDIR")
    if [ ! -w "$outdir_parent" ]; then
      echo -e "${RED}❌ No write permission in output directory parent: $outdir_parent${NC}"
      exit 1
    fi

    mapfile -t VAL_FILES < <(find "$VAL_DIR" -maxdepth 1 -type f -iname "*.tif" | sort)
    if [ "${#VAL_FILES[@]}" -eq 0 ]; then
        echo -e "${RED}❌ No .tif files found in $VAL_DIR!${NC}"
        exit 1
    fi

    echo -e "${YELLOW}📂 Found ${#VAL_FILES[@]} validation movies:${NC}"
    for v in "${VAL_FILES[@]}"; do
        echo "   - $v"
    done

    RUN_SUMMARY=""
    OUTDIRS=()
    for VAL_MOVIE in "${VAL_FILES[@]}"; do
        BASENAME=$(basename "$VAL_MOVIE" .tif)
        MODEL_ID="$(basename "$INPUT_TRAIN" .tif)_${BACKBONE}_${BASENAME}_$TIMESTAMP"
        CURR_OUTDIR="${OUTDIR%/}_${MODEL_ID}"
        OUTDIRS+=("$CURR_OUTDIR")
        mkdir -p "$CURR_OUTDIR" || { echo -e "${RED}❌ Cannot create output directory: $CURR_OUTDIR${NC}"; exit 1; }
        mkdir -p "$CURR_OUTDIR/figures"
        MODEL_RUN_DIR="runs/${MODEL_ID}"
        mkdir -p "$MODEL_RUN_DIR" || { echo -e "${RED}❌ Cannot create model run directory: $MODEL_RUN_DIR${NC}"; exit 1; }
        CONFIG_FILE="$CURR_OUTDIR/run_config.yaml"

        cat <<EOL > "$CONFIG_FILE"
name: $MODEL_ID
epochs: 0
augment: 0
batchsize: 108
size: $CROP_SIZE
cam_size: 960
backbone: $BACKBONE
features: 32
train_samples_per_epoch: 50000
num_workers: 4
projhead: minimal_batchnorm
classhead: none
input_train:
  - "$INPUT_TRAIN"
input_val:
  - "$VAL_MOVIE"
input_mask:
  - ""
split_train:
  - [0.0, 1.0]
split_val:
  - [0.0, 1.0]
outdir: "$MODEL_RUN_DIR"
gpu: "0"
seed: $SEED
pixel_resolution: $PIXEL_RES
tensorboard: false
write_final_cams: true
binarize: false
min_pixels: $MIN_PIXELS
config_yaml: "$CONFIG_FILE"
EOL

        center_text "${BLUE}🚀 TAP-only eval for $BASENAME${NC}"
        LOGFILE="$CURR_OUTDIR/pipeline_log.txt"
        METRICS_CSV="$CURR_OUTDIR/pipeline_metrics.csv"
        echo "step_name,elapsed_sec,peak_ram_mb,disk_after_mb" > "$METRICS_CSV"

        (
          exec > >(tee -i "$LOGFILE")
          exec 2>&1

          STEP_LOG="$CURR_OUTDIR/02_dataprep_timing.log"
          START_TIME=$(date +%s)
          run_and_log "Data Preparation" "$STEP_LOG" "$CURR_OUTDIR" "$METRICS_CSV" python3 Workflow/02_data_prep.py \
            --input_frame "$VAL_MOVIE" \
            --input_mask "" \
            --data_save_dir "$CURR_OUTDIR" \
            --size "$CROP_SIZE" \
            --binarize \
            --min_pixels "$MIN_PIXELS" \
            --data_seed "$SEED"
          END_TIME=$(date +%s)
          RUNTIME=$((END_TIME - START_TIME))
          echo "[$BASENAME] Completed in $((RUNTIME/60))m $((RUNTIME% 60))s"
        )
        if [[ $? -ne 0 ]]; then
          RUN_SUMMARY+="\n🔸 $BASENAME: FAILED!"
        else
          RUN_SUMMARY+="\n🔸 $BASENAME: SUCCESS. Output Dir: $CURR_OUTDIR"
        fi
    done

    center_text "${GREEN}🎉 Batch TAP Validation Complete!${NC}"
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"
    echo "🔹 Training movie: $INPUT_TRAIN"
    echo "🔹 Crop Size     : $CROP_SIZE"
    echo "🔹 Pixel Res     : $PIXEL_RES"
    echo "🔹 Backbone      : $BACKBONE"
    echo "🔹 Random Seed   : $SEED"
    echo "🔹 Validation dir: $VAL_DIR"
    echo -e "🔹 Results: $RUN_SUMMARY"
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"

    center_text "${YELLOW}📝 Generating HTML Report (Batch Mode)${NC}"
    python3 Workflow/05_generate_report.py --batch_outdirs "${OUTDIRS[@]}"

    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${GREEN}🎯 CellFate BATCH RESULTS SUMMARY${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}🔸 Output directories:${NC}"
    for d in "${OUTDIRS[@]}"; do
        echo -e "   $d"
        if [ -f "$d/report.html" ]; then
            echo -e "     ↳ ${GREEN}Report:${NC} $d/report.html"
        fi
        if [ -f "$d/pipeline_log.txt" ]; then
            echo -e "     ↳ ${GREEN}Log:   ${NC} $d/pipeline_log.txt"
        fi
        if [ -f "$d/pipeline_metrics.csv" ]; then
            echo -e "     ↳ ${GREEN}Metrics CSV: ${NC} $d/pipeline_metrics.csv"
        fi
        if [ -d "$d/figures" ]; then
            echo -e "     ↳ ${GREEN}Figures:     ${NC} $d/figures"
        fi
    done
    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${GREEN}Open any report in your browser, e.g.:${NC} file://[OUTPUTDIR]/report.html"
fi
