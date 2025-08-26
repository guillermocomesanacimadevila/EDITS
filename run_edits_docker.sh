#!/bin/bash

# ──────────────────────────────────────────────────────────────── #
#                 CELLFLOW PIPELINE: DOCKER VERSION               #
#        (Docker auto-check + environment bootstrap + pipeline)   #
# ──────────────────────────────────────────────────────────────── #

# ───────────── Terminal Colors ───────────── #
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ───────────── Help Option ───────────── #
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo -e "${YELLOW}CELLFLOW PIPELINE (Docker Version)${NC}"
    echo -e "Usage: bash $0"
    echo "You will be interactively prompted for input files/parameters."
    echo "Outputs go to ./runs/ and to your specified output directory."
    echo -e "After run, open your HTML report in your browser.\n"
    exit 0
fi

# ───────────── Ctrl+C Trap with Cleanup ───────────── #
trap 'echo -e "\n${RED}⚡️ Script interrupted by user. Exiting!${NC}"; exit 1' SIGINT

# ───────────── Check Required Commands ───────────── #
REQUIRED_COMMANDS=(docker tee)
for cmd in "${REQUIRED_COMMANDS[@]}"; do
  if ! command -v "$cmd" &> /dev/null; then
    echo -e "${RED}❌ Required command '$cmd' not found. Please install it before running this script.${NC}"
    exit 1
  fi
done

# ───────────── Docker Running Check (Auto-start if possible) ───────────── #
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}⏳ Docker is not running.${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${BLUE}🚀 Attempting to start Docker Desktop on Mac...${NC}"
        open -a Docker
        echo -e "${YELLOW}⌛ Waiting for Docker to launch...${NC}"
        while ! docker info > /dev/null 2>&1; do sleep 2; done
        echo -e "${GREEN}✅ Docker is running.${NC}"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo -e "${BLUE}🚀 Attempting to start Docker service on Linux...${NC}"
        sudo systemctl start docker
        sleep 5
        if ! docker info > /dev/null 2>&1; then
            echo -e "${RED}❌ Docker is still not running. Please start it manually.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✅ Docker is running.${NC}"
    else
        echo -e "${YELLOW}⚠️  Please start Docker Desktop manually!${NC}"
        read -p "Press [Enter] when Docker is running..."
        if ! docker info > /dev/null 2>&1; then
            echo -e "${RED}❌ Docker is still not running. Exiting.${NC}"
            exit 1
        fi
    fi
fi

# ───────────── Disk Space Check (500MB) ───────────── #
REQUIRED_SPACE_MB=500
AVAILABLE_SPACE_KB=$(df "$HOME" | tail -1 | awk '{print $4}')
AVAILABLE_SPACE_MB=$((AVAILABLE_SPACE_KB / 1024))
if (( AVAILABLE_SPACE_MB < REQUIRED_SPACE_MB )); then
  echo -e "${RED}❌ Not enough disk space: ${AVAILABLE_SPACE_MB}MB available, ${REQUIRED_SPACE_MB}MB needed.${NC}"
  exit 1
fi

# ───────────── Version Logging ───────────── #
echo -e "${YELLOW}🔢 Environment Versions (Docker host):${NC}"
echo -n "Docker: "; docker --version
echo "----------------------------"

# ───────────── Data Directory & Runs Directory ───────────── #
if [ ! -d "Data" ]; then
    echo -e "${RED}❌ 'Data' directory does not exist in the project root! Please create it and add your data files.${NC}"
    exit 1
fi
[ ! -d "runs" ] && mkdir runs
if [ ! -w "$HOME" ]; then
    echo -e "${RED}❌ No write permission to home directory: $HOME${NC}"
    exit 1
fi
if [ ! -w "runs" ]; then
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

# ───────────── File/Directory Selection Functions ───────────── #
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

center_text() {
    local width=70
    local text="$1"
    printf "\n%*s\n\n" $(( (${#text} + width) / 2 )) "$text"
}

# ──────────────────────────────────────────────────────────────── #
#                           USER INPUT                            #
# ──────────────────────────────────────────────────────────────── #
center_text "${BLUE}🔬 CELLFLOW ML Pipeline Setup${NC}"
echo -e "${YELLOW}ℹ️  Select files/directories interactively below (relative paths preferred)${NC}"

INPUT_TRAIN=$(select_file "📥 Select TRAINING movie (.tif)" "Data/")

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
    INPUT_VAL=$(select_file "🧪 Select VALIDATION movie (.tif)" "Data/")
    INPUT_MASK=$(select_file "🎭 Select ANNOTATED MASK (.tif)" "Data/")
    read -p "$(center_text '📐 Crop size (e.g., 48):')" CROP_SIZE
    read -p "$(center_text '🔬 Pixel resolution (e.g., 0.65):')" PIXEL_RES
    read -p "$(center_text '🔁 Number of training epochs:')" EPOCHS
    read -p "$(center_text '📂 Output directory path (e.g., Data/toy_data):')" OUTDIR
    read -p "$(center_text '🎲 Random seed:')" SEED
    read -p "$(center_text '🧠 Backbone (unet, spectformer-xs):')" BACKBONE
    read -p "$(center_text '🔸 Minimum # pixels in event mask to count as event (min_pixels, e.g., 10):')" MIN_PIXELS

    # Input validation
    re_int='^[0-9]+$'
    re_float='^[0-9]+(\.[0-9]+)?$'
    [ ! -f "$INPUT_TRAIN" ] && echo -e "${RED}❌ Training movie not found at '$INPUT_TRAIN'${NC}" && exit 1
    [ ! -f "$INPUT_VAL" ] && echo -e "${RED}❌ Validation movie not found at '$INPUT_VAL'${NC}" && exit 1
    [ ! -f "$INPUT_MASK" ] && echo -e "${RED}❌ Mask not found at '$INPUT_MASK'${NC}" && exit 1
    [[ ! "$CROP_SIZE" =~ $re_int ]] && echo -e "${RED}❌ Crop size must be integer.${NC}" && exit 1
    [[ ! "$EPOCHS" =~ $re_int ]] && echo -e "${RED}❌ Epochs must be integer.${NC}" && exit 1
    [[ ! "$SEED" =~ $re_int ]] && echo -e "${RED}❌ Seed must be integer.${NC}" && exit 1
    [[ ! "$PIXEL_RES" =~ $re_float ]] && echo -e "${RED}❌ Pixel resolution must be float.${NC}" && exit 1
    [[ "$BACKBONE" != "unet" && "$BACKBONE" != "spectformer-xs" ]] && echo -e "${RED}❌ Unsupported backbone: $BACKBONE${NC}" && exit 1
    [[ ! "$MIN_PIXELS" =~ $re_int ]] && echo -e "${RED}❌ min_pixels must be integer.${NC}" && exit 1

    # Check output directory write permission before mkdir
    outdir_parent=$(dirname "$OUTDIR")
    if [ ! -w "$outdir_parent" ]; then
      echo -e "${RED}❌ No write permission in output directory parent: $outdir_parent${NC}"
      exit 1
    fi

    # Automatic ID: movie name + backbone + timestamp
    MODEL_ID="$(basename "$INPUT_TRAIN" .tif)_${BACKBONE}_$TIMESTAMP"
    MODEL_RUN_DIR="runs/${MODEL_ID}"
    mkdir -p "$MODEL_RUN_DIR" || { echo -e "${RED}❌ Cannot create model run directory: $MODEL_RUN_DIR${NC}"; exit 1; }

    OUTDIR="${OUTDIR%/}_${MODEL_ID}"
    mkdir -p "$OUTDIR" || { echo -e "${RED}❌ Cannot create output directory: $OUTDIR${NC}"; exit 1; }

    # Map all file paths for the Docker container
    CONTAINER_TRAIN="/app/$INPUT_TRAIN"
    CONTAINER_VAL="/app/$INPUT_VAL"
    CONTAINER_MASK="/app/$INPUT_MASK"
    CONTAINER_OUTDIR="/app/$OUTDIR"
    CONTAINER_RUNSDIR="/app/$MODEL_RUN_DIR"
    CONTAINER_CONFIG="$CONTAINER_OUTDIR/run_config.yaml"

    # Save config file (still used for record-keeping & report)
    CONFIG_FILE="$OUTDIR/run_config.yaml"
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
classhead: minimal
input_train:
  - "$CONTAINER_TRAIN"
input_val:
  - "$CONTAINER_VAL"
input_mask:
  - "$CONTAINER_MASK"
split_train:
  - [0.0, 1.0]
split_val:
  - [0.0, 1.0]
outdir: "$CONTAINER_RUNSDIR"
gpu: "0"
seed: $SEED
pixel_resolution: $PIXEL_RES
tensorboard: true
write_final_cams: false
binarize: false
min_pixels: $MIN_PIXELS
config_yaml: "$CONTAINER_CONFIG"
EOL

    center_text "${GREEN}📝 Configuration saved to $CONFIG_FILE${NC}"

    # Logging setup
    LOGFILE="$OUTDIR/pipeline_log.txt"
    exec > >(tee -i "$LOGFILE")
    exec 2>&1

    START_TIME=$(date +%s)

    center_text "${YELLOW}🚀 Training Model (Fine-tuning)${NC}"
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
      python Workflow/01_fine-tune.py --config "$CONTAINER_CONFIG" || { echo -e "${RED}❌ Fine-tuning failed!${NC}"; exit 1; }

    center_text "${YELLOW}🚀 Data Preparation${NC}"
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
      python Workflow/02_data_prep.py \
        --input_frame "$CONTAINER_VAL" \
        --input_mask "$CONTAINER_MASK" \
        --data_save_dir "$CONTAINER_OUTDIR" \
        --size "$CROP_SIZE" \
        --pixel_area_threshold 0 \
        --binarize \
        --min_pixels "$MIN_PIXELS" \
        --data_seed "$SEED" \
        || { echo -e "${RED}❌ Data prep failed!${NC}"; exit 1; }

    # ───────────── Event Classification ───────────── #
    TAP_MODEL_DIR="${MODEL_RUN_DIR}/${MODEL_ID}_backbone_${BACKBONE}"
    echo "TAP model folder: $TAP_MODEL_DIR"
    ls -lh "$TAP_MODEL_DIR" || echo "Model dir not yet populated (will be after 01_fine-tune.py)"

    center_text "${YELLOW}🚀 Event Classification${NC}"
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
      python Workflow/03_event_classification.py \
        --input_frame "$CONTAINER_VAL" \
        --input_mask "$CONTAINER_MASK" \
        --cam_size 960 \
        --size "$CROP_SIZE" \
        --batchsize 108 \
        --training_epochs "$EPOCHS" \
        --balanced_sample_size 50000 \
        --crops_per_image 108 \
        --model_seed "$SEED" \
        --data_seed "$SEED" \
        --data_save_dir "$CONTAINER_OUTDIR" \
        --num_runs 1 \
        --model_save_dir "$CONTAINER_RUNSDIR" \
        --model_id "$MODEL_ID" \
        --cls_head_arch linear \
        --backbone "$BACKBONE" \
        --name "$MODEL_ID" \
        --binarize false \
        --TAP_model_load_path "/app/$TAP_MODEL_DIR" \
        || { echo -e "${RED}❌ Classification failed!${NC}"; exit 1; }

    center_text "${YELLOW}🚀 Examining Mistaken Predictions${NC}"
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
      python Workflow/04_examine_mistaken_predictions.py \
        --mistake_pred_dir "$CONTAINER_RUNSDIR" \
        --masks_path "$CONTAINER_MASK" \
        --TAP_model_load_path "/app/$TAP_MODEL_DIR" \
        --patch_size "$CROP_SIZE" \
        --test_data_load_path "$CONTAINER_OUTDIR/test_data_crops_flat.pth" \
        --combined_model_load_dir "$CONTAINER_RUNSDIR" \
        --model_id "$MODEL_ID" \
        --cls_head_arch linear \
        --num_egs_to_show 10 \
        --save_data \
        || { echo -e "${RED}❌ Mistake analysis failed!${NC}"; exit 1; }

    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    # ───────────── Pipeline Summary ───────────── #
    center_text "${GREEN}🎉 CELLFLOW Pipeline Complete!${NC}"
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"
    echo "🔹 Model ID      : $MODEL_ID"
    echo "🔹 Model Dir     : $MODEL_RUN_DIR"
    echo "🔹 Output Dir    : $OUTDIR"
    echo "🔹 Crop Size     : $CROP_SIZE"
    echo "🔹 Epochs        : $EPOCHS"
    echo "🔹 Pixel Res     : $PIXEL_RES"
    echo "🔹 Backbone      : $BACKBONE"
    echo "🔹 Mask File     : $INPUT_MASK"
    echo "🔹 Config File   : $CONFIG_FILE"
    echo "🔹 Log File      : $LOGFILE"
    echo "⏱️  Runtime: $((RUNTIME / 60)) min $((RUNTIME % 60)) sec"
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"

    center_text "${YELLOW}📝 Generating HTML Report${NC}"
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
      python Workflow/05_generate_report.py --config "$CONTAINER_CONFIG" --outdir "$CONTAINER_OUTDIR"
    echo -e "${GREEN}📄 Report generated at $OUTDIR/report.html${NC}"

else
    # ───────────── BATCH MODE ───────────── #
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

    # Check output directory write permission before mkdir
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
        MODEL_RUN_DIR="runs/${MODEL_ID}"
        mkdir -p "$MODEL_RUN_DIR" || { echo -e "${RED}❌ Cannot create model run directory: $MODEL_RUN_DIR${NC}"; exit 1; }
        CONFIG_FILE="$CURR_OUTDIR/run_config.yaml"

        # Container path mappings
        CONTAINER_TRAIN="/app/$INPUT_TRAIN"
        CONTAINER_VAL="/app/$VAL_MOVIE"
        CONTAINER_OUTDIR="/app/$CURR_OUTDIR"
        CONTAINER_RUNSDIR="/app/$MODEL_RUN_DIR"
        CONTAINER_CONFIG="$CONTAINER_OUTDIR/run_config.yaml"

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
  - "$CONTAINER_TRAIN"
input_val:
  - "$CONTAINER_VAL"
input_mask:
  - ""
split_train:
  - [0.0, 1.0]
split_val:
  - [0.0, 1.0]
outdir: "$CONTAINER_RUNSDIR"
gpu: "0"
seed: $SEED
pixel_resolution: $PIXEL_RES
tensorboard: false
write_final_cams: true
binarize: false
min_pixels: $MIN_PIXELS
config_yaml: "$CONTAINER_CONFIG"
EOL

        center_text "${BLUE}🚀 TAP-only eval for $BASENAME${NC}"
        LOGFILE="$CURR_OUTDIR/pipeline_log.txt"
        (
          exec > >(tee -i "$LOGFILE")
          exec 2>&1

          START_TIME=$(date +%s)
          docker run --rm -v "$PWD":/app tap_pipeline:latest \
            python Workflow/02_data_prep.py \
              --input_frame "$CONTAINER_VAL" \
              --input_mask "" \
              --data_save_dir "$CONTAINER_OUTDIR" \
              --size "$CROP_SIZE" \
              --pixel_area_threshold 0 \
              --binarize \
              --min_pixels "$MIN_PIXELS" \
              --data_seed "$SEED" \
              || exit 99
          END_TIME=$(date +%s)
          RUNTIME=$((END_TIME - START_TIME))
          echo "[$BASENAME] Completed in $((RUNTIME/60))m $((RUNTIME%60))s"
        )
        if [[ $? -eq 99 ]]; then
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
    # Space-separated output directories, all in /app inside container
    CONTAINER_OUTDIRS=""
    for o in "${OUTDIRS[@]}"; do
        CONTAINER_OUTDIRS+=" /app/$o"
    done
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
        python Workflow/05_generate_report.py --batch_outdirs $CONTAINER_OUTDIRS
    echo -e "${GREEN}📄 Batch report(s) generated.${NC}"
fi
