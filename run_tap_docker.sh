#!/bin/bash

# ──────────────────────────────────────────────────────────────── #
#                        CELLFLOW PIPELINE                        #
#                      Docker-Only Version                        #
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

# ───────────── Ctrl+C Trap ───────────── #
trap 'echo -e "\n${RED}⚡️ Script interrupted by user. Exiting!${NC}"; exit 1' SIGINT

# ──────────────────────────────────────────────────────────────── #
#            AUTO-START DOCKER IF NOT RUNNING                     #
# ──────────────────────────────────────────────────────────────── #
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

# ───────────── Version Logging ───────────── #
echo -e "${YELLOW}🔢 Environment Versions (Docker host):${NC}"
echo -n "Docker: "; docker --version
echo "----------------------------"

# ──────────────────────────────────────────────────────────────── #
#                CHECK DATA DIR AND RUNS DIR                      #
# ──────────────────────────────────────────────────────────────── #
if [ ! -d "Data" ]; then
    echo -e "${RED}❌ 'Data' directory does not exist in the project root! Please create it and add your data files.${NC}"
    exit 1
fi
[ ! -d "runs" ] && mkdir runs

# ──────────────────────────────────────────────────────────────── #
#                           USER INPUT                            #
# ──────────────────────────────────────────────────────────────── #
center_text() {
    local width=70
    local text="$1"
    printf "\n%*s\n\n" $(( (${#text} + width) / 2 )) "$text"
}

center_text "${BLUE}🔬 CELLFLOW ML Pipeline Setup${NC}"
echo -e "${YELLOW}ℹ️  Provide paths relative to the project root (e.g. Data/toy_data/toy_movie.tif)${NC}"

read -p "$(center_text '📥 Path to training movie (.tif, e.g. Data/toy_data/toy_movie.tif):')" INPUT_TRAIN

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
    read -p "$(center_text '🧪 Path to validation movie (.tif):')" INPUT_VAL
    read -p "$(center_text '🎭 Path to annotated mask (.tif):')" INPUT_MASK
    read -p "$(center_text '📐 Crop size (e.g., 48):')" CROP_SIZE
    read -p "$(center_text '🔬 Pixel resolution (e.g., 0.65):')" PIXEL_RES
    read -p "$(center_text '🔁 Number of training epochs:')" EPOCHS
    read -p "$(center_text '📂 Output directory path (e.g., Data/toy_data):')" OUTDIR
    read -p "$(center_text '🎲 Random seed:')" SEED
    read -p "$(center_text '🧠 Backbone (unet, spectformer-xs):')" BACKBONE

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

    # Automatic ID: movie name + backbone + timestamp
    MODEL_ID="$(basename "$INPUT_TRAIN" .tif)_${BACKBONE}_$TIMESTAMP"
    MODEL_RUN_DIR="runs/${MODEL_ID}"
    mkdir -p "$MODEL_RUN_DIR"
    OUTDIR="${OUTDIR%/}_${MODEL_ID}"
    mkdir -p "$OUTDIR"

    # Paths for Docker container mapping
    CONTAINER_TRAIN="/data/${INPUT_TRAIN#Data/}"
    CONTAINER_VAL="/data/${INPUT_VAL#Data/}"
    CONTAINER_MASK="/data/${INPUT_MASK#Data/}"
    CONTAINER_OUTDIR="/app/${OUTDIR}"
    CONTAINER_RUNSDIR="/app/runs/${MODEL_ID}"

    # Save config file
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
config_yaml: "$CONTAINER_OUTDIR/run_config.yaml"
EOL

    center_text "${GREEN}📝 Configuration saved to $CONFIG_FILE${NC}"

    LOGFILE="$OUTDIR/pipeline_log.txt"
    exec > >(tee -i "$LOGFILE")
    exec 2>&1

    START_TIME=$(date +%s)

    center_text "${YELLOW}🚀 Training Model (Fine-tuning)${NC}"
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
      python Workflow/01_fine-tune.py --config "$CONTAINER_OUTDIR/run_config.yaml" || { echo -e "${RED}❌ Fine-tuning failed!${NC}"; exit 1; }

    center_text "${YELLOW}🚀 Data Preparation${NC}"
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
      python Workflow/02_data_prep.py --config "$CONTAINER_OUTDIR/run_config.yaml" || { echo -e "${RED}❌ Data prep failed!${NC}"; exit 1; }

    center_text "${YELLOW}🚀 Event Classification${NC}"
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
      python Workflow/03_event_classification.py --config "$CONTAINER_OUTDIR/run_config.yaml" || { echo -e "${RED}❌ Classification failed!${NC}"; exit 1; }

    center_text "${YELLOW}🚀 Examining Mistaken Predictions${NC}"
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
      python Workflow/04_examine_mistaken_predictions.py --config "$CONTAINER_OUTDIR/run_config.yaml" || { echo -e "${RED}❌ Mistake analysis failed!${NC}"; exit 1; }

    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

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
      python Workflow/05_generate_report.py --config "$CONTAINER_OUTDIR/run_config.yaml" --outdir "$CONTAINER_OUTDIR"
    echo -e "${GREEN}📄 Report generated at $OUTDIR/report.html${NC}"

else
    # Batch mode - only runs Data Prep
    read -p "$(center_text '📂 Directory with validation .tif files:')" VAL_DIR
    read -p "$(center_text '📐 Crop size (e.g., 48):')" CROP_SIZE
    read -p "$(center_text '🔬 Pixel resolution (e.g., 0.65):')" PIXEL_RES
    read -p "$(center_text '📂 Output directory path (e.g., Data/toy_data):')" OUTDIR
    read -p "$(center_text '🎲 Random seed:')" SEED
    read -p "$(center_text '🧠 Backbone (unet, spectformer-xs):')" BACKBONE

    re_int='^[0-9]+$'
    re_float='^[0-9]+(\.[0-9]+)?$'
    [ ! -f "$INPUT_TRAIN" ] && echo -e "${RED}❌ Training movie not found at '$INPUT_TRAIN'${NC}" && exit 1
    [ ! -d "$VAL_DIR" ] && echo -e "${RED}❌ Validation directory not found at '$VAL_DIR'${NC}" && exit 1
    [[ ! "$CROP_SIZE" =~ $re_int ]] && echo -e "${RED}❌ Crop size must be integer.${NC}" && exit 1
    [[ ! "$SEED" =~ $re_int ]] && echo -e "${RED}❌ Seed must be integer.${NC}" && exit 1
    [[ ! "$PIXEL_RES" =~ $re_float ]] && echo -e "${RED}❌ Pixel resolution must be float.${NC}" && exit 1
    [[ "$BACKBONE" != "unet" && "$BACKBONE" != "spectformer-xs" ]] && echo -e "${RED}❌ Unsupported backbone: $BACKBONE${NC}" && exit 1

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
        mkdir -p "$CURR_OUTDIR"
        MODEL_RUN_DIR="runs/${MODEL_ID}"
        mkdir -p "$MODEL_RUN_DIR"
        CONFIG_FILE="$CURR_OUTDIR/run_config.yaml"
        CONTAINER_TRAIN="/data/${INPUT_TRAIN#Data/}"
        CONTAINER_VAL="/data/${VAL_MOVIE#Data/}"
        CONTAINER_OUTDIR="/app/${CURR_OUTDIR}"
        CONTAINER_RUNSDIR="/app/runs/${MODEL_ID}"

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
config_yaml: "$CONTAINER_OUTDIR/run_config.yaml"
EOL

        center_text "${BLUE}🚀 TAP-only eval for $BASENAME${NC}"
        LOGFILE="$CURR_OUTDIR/pipeline_log.txt"
        exec > >(tee -i "$LOGFILE")
        exec 2>&1

        START_TIME=$(date +%s)
        docker run --rm -v "$PWD":/app tap_pipeline:latest \
            python Workflow/02_data_prep.py --config "$CONTAINER_OUTDIR/run_config.yaml" \
            || { echo -e "${RED}❌ TAP-only data prep failed for $BASENAME!${NC}"; RUN_SUMMARY+="\n🔸 $BASENAME: FAILED!"; continue; }
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))
        RUN_SUMMARY+="\n🔸 $BASENAME: SUCCESS. Output Dir: $CURR_OUTDIR (Runtime: $((RUNTIME / 60)) min $((RUNTIME % 60)) sec)"
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
    docker run --rm -v "$PWD":/app tap_pipeline:latest \
        python Workflow/05_generate_report.py --batch_outdirs "${OUTDIRS[@]}"
    echo -e "${GREEN}📄 Batch report(s) generated.${NC}"

fi
