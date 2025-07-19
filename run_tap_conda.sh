#!/bin/bash

# ──────────────────────────────────────────────────────────────── #
#                 CELLFLOW PIPELINE: SELF-CONFIGURING             #
#      (Conda auto-install + environment bootstrap + pipeline)     #
# ──────────────────────────────────────────────────────────────── #

# ───────────── Terminal Colors ───────────── #
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ───────────── Locale (for font/Unicode issues) ───────────── #
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# ───────────── Help Option ───────────── #
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo -e "${YELLOW}CELLFLOW PIPELINE${NC}"
    echo -e "Usage: bash $0"
    echo "You will be interactively prompted for input files/parameters."
    echo "Outputs go to ./runs/ and to your specified output directory."
    echo -e "After run, open your HTML report in your browser.\n"
    exit 0
fi

# ───────────── Ctrl+C Trap with Cleanup ───────────── #
trap 'echo -e "\n${RED}⚡️ Script interrupted by user. Exiting!${NC}"; exit 1' SIGINT

# ──────────────────────────────────────────────────────────────── #
#      CONDA/AUTOINSTALL/ENV CREATION                             #
# ──────────────────────────────────────────────────────────────── #
ENV_NAME="cellflow-env"
ENV_YML="environment.yml"

if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}🔄 Conda not found. Installing Miniconda...${NC}"
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    export PATH="$HOME/miniconda/bin:$PATH"
    source "$HOME/miniconda/etc/profile.d/conda.sh"
    echo -e "${GREEN}✅ Miniconda installed.${NC}"
else
    eval "$(conda shell.bash hook)"
fi

if ! conda env list | grep -qw "$ENV_NAME"; then
    echo -e "${YELLOW}🔧 Creating Conda env '$ENV_NAME' from $ENV_YML...${NC}"
    if [ ! -f "$ENV_YML" ]; then
        echo -e "${RED}❌ $ENV_YML not found! Cannot create conda env.${NC}"
        exit 1
    fi
    conda env create -f "$ENV_YML" -n "$ENV_NAME"
    echo -e "${GREEN}✅ Conda environment '$ENV_NAME' created.${NC}"
fi

echo -e "${GREEN}🔄 Activating '$ENV_NAME'...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME || { echo -e "${RED}❌ Failed to activate '$ENV_NAME'!${NC}"; exit 1; }

# ───────────── FONT FIX FOR MATPLOTLIB ───────────── #
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y fonts-dejavu-core fontconfig
fi

mkdir -p ~/.config/matplotlib
echo "font.family: sans-serif
font.sans-serif: DejaVu Sans
" > ~/.config/matplotlib/matplotlibrc

# ───────────── TAP/tarrow install (editable mode) ───────────── #
echo -e "${YELLOW}🔗 Installing TAP/tarrow package in editable mode (if not already)...${NC}"
if [ -d "TAP/tarrow" ] && [ -f "TAP/tarrow/setup.py" ]; then
    pip show tarrow > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        pip install -e TAP/tarrow
    fi
else
    echo -e "${RED}❌ TAP/tarrow directory or setup.py not found!${NC}"
    exit 1
fi

# ───────────── Version Logging ───────────── #
echo -e "${YELLOW}🔢 Environment Versions:${NC}"
echo -n "Python: "; python --version
echo -n "Conda: "; conda --version
echo -n "PyTorch: "; python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "N/A"
echo -n "Numpy: "; python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo "N/A"
echo "----------------------------"

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
#                           USER INPUT                            #
# ──────────────────────────────────────────────────────────────── #
center_text() {
    local width=70
    local text="$1"
    printf "\n%*s\n\n" $(( (${#text} + width) / 2 )) "$text"
}

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
    # ──────────────────────────────────────────────────────────────── #
    #                 SINGLE VALIDATION MOVIE (FULL PIPELINE)         #
    # ──────────────────────────────────────────────────────────────── #
    INPUT_VAL=$(select_file "🧪 Select VALIDATION movie (.tif)" "Data/")
    INPUT_MASK=$(select_file "🎭 Select ANNOTATED MASK (.tif)" "Data/")
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

    # ──────────────────────────────────────────────────────────────── #
    #                        SAVE CONFIG FILE                         #
    # ──────────────────────────────────────────────────────────────── #
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
seed: $SEED
pixel_resolution: $PIXEL_RES
tensorboard: true
write_final_cams: false
binarize: false
config_yaml: "$CONFIG_FILE"
EOL

    center_text "${GREEN}📝 Configuration saved to $CONFIG_FILE${NC}"

    # ──────────────────────────────────────────────────────────────── #
    #                       RUN FULL PIPELINE                         #
    # ──────────────────────────────────────────────────────────────── #
    LOGFILE="$OUTDIR/pipeline_log.txt"
    exec > >(tee -i "$LOGFILE")
    exec 2>&1

    START_TIME=$(date +%s)

    center_text "${YELLOW}🚀 Training Model (Fine-tuning)${NC}"
    python Workflow/01_fine-tune.py --config "$CONFIG_FILE" || { echo -e "${RED}❌ Fine-tuning failed!${NC}"; exit 1; }

    center_text "${YELLOW}🚀 Data Preparation${NC}"
    python Workflow/02_data_prep.py \
        --input_frame "$INPUT_VAL" \
        --input_mask "$INPUT_MASK" \
        --data_save_dir "$OUTDIR" \
        --size "$CROP_SIZE" \
        --pixel_area_threshold 0 \
        --binarize \
        --data_seed "$SEED" \
        || { echo -e "${RED}❌ Data prep failed!${NC}"; exit 1; }

    # ──────────────────────────────────────────────────────────────── #
    #                EVENT CLASSIFICATION (TAP MODEL FIX)             #
    # ──────────────────────────────────────────────────────────────── #
    TAP_MODEL_DIR="${MODEL_RUN_DIR}/${MODEL_ID}_backbone_${BACKBONE}"
    echo "TAP model folder: $TAP_MODEL_DIR"
    ls -lh "$TAP_MODEL_DIR"

    center_text "${YELLOW}🚀 Event Classification${NC}"
    python Workflow/03_event_classification.py \
        --input_frame "$INPUT_VAL" \
        --input_mask "$INPUT_MASK" \
        --cam_size 960 \
        --size "$CROP_SIZE" \
        --batchsize 108 \
        --training_epochs "$EPOCHS" \
        --balanced_sample_size 50000 \
        --crops_per_image 108 \
        --model_seed "$SEED" \
        --data_seed "$SEED" \
        --data_save_dir "$OUTDIR" \
        --num_runs 1 \
        --model_save_dir "$MODEL_RUN_DIR" \
        --model_id "$MODEL_ID" \
        --cls_head_arch linear \
        --backbone "$BACKBONE" \
        --name "$MODEL_ID" \
        --binarize false \
        --TAP_model_load_path "$TAP_MODEL_DIR" \
        || { echo -e "${RED}❌ Classification failed!${NC}"; exit 1; }

    center_text "${YELLOW}🚀 Examining Mistaken Predictions${NC}"
    python Workflow/04_examine_mistaken_predictions.py \
        --mistake_pred_dir "$MODEL_RUN_DIR" \
        --masks_path "$INPUT_MASK" \
        --TAP_model_load_path "$TAP_MODEL_DIR" \
        --patch_size "$CROP_SIZE" \
        --test_data_load_path "$OUTDIR/test_data_crops_flat.pth" \
        --combined_model_load_dir "$MODEL_RUN_DIR" \
        --model_id "$MODEL_ID" \
        --cls_head_arch linear \
        --num_egs_to_show 10 \
        --save_data \
        || { echo -e "${RED}❌ Mistake analysis failed!${NC}"; exit 1; }

    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    # ──────────────────────────────────────────────────────────────── #
    #         [REMOVED] GENERATE FIGURES SECTION (was 06_generate_figures.py)
    # ──────────────────────────────────────────────────────────────── #
    # center_text "${YELLOW}📊 Generating Publication-Ready Figures${NC}"
    # python Workflow/06_generate_figures.py --config "$CONFIG_FILE" --outdir "$OUTDIR"

    # ──────────────────────────────────────────────────────────────── #
    #                              SUMMARY                            #
    # ──────────────────────────────────────────────────────────────── #
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

    # ──────────────────────────────────────────────────────────────── #
    #                   GENERATE HTML REPORT (FULL)                   #
    # ──────────────────────────────────────────────────────────────── #
    center_text "${YELLOW}📝 Generating HTML Report${NC}"
    python Workflow/05_generate_report.py --config "$CONFIG_FILE" --outdir "$OUTDIR"

    # ──────────────────────────────────────────────────────────────── #
    #                       RESULTS SUMMARY                           #
    # ──────────────────────────────────────────────────────────────── #
    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${GREEN}🎯 CELLFLOW RESULTS SUMMARY${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}🔸 Output directory    :${NC} $OUTDIR"
    echo -e "${YELLOW}🔸 Log file           :${NC} $LOGFILE"
    echo -e "${YELLOW}🔸 Config file        :${NC} $CONFIG_FILE"
    echo -e "${YELLOW}🔸 HTML report        :${NC} $OUTDIR/report.html"
    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${GREEN}Open your report in your browser:${NC} file://$OUTDIR/report.html"
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
config_yaml: "$CONFIG_FILE"
EOL

        center_text "${BLUE}🚀 TAP-only eval for $BASENAME${NC}"
        LOGFILE="$CURR_OUTDIR/pipeline_log.txt"
        (
          exec > >(tee -i "$LOGFILE")
          exec 2>&1

          START_TIME=$(date +%s)
          python Workflow/02_data_prep.py \
            --input_frame "$VAL_MOVIE" \
            --input_mask "" \
            --data_save_dir "$CURR_OUTDIR" \
            --size "$CROP_SIZE" \
            --pixel_area_threshold 0 \
            --binarize \
            --data_seed "$SEED" \
            || { echo -e "${RED}❌ TAP-only data prep failed for $BASENAME!${NC}"; exit 99; }
          END_TIME=$(date +%s)
          RUNTIME=$((END_TIME - START_TIME))
          echo "[$BASENAME] Completed in $((RUNTIME/60))m $((RUNTIME% 60))s"
        )
        if [[ $? -eq 99 ]]; then
          RUN_SUMMARY+="\n🔸 $BASENAME: FAILED!"
        else
          RUN_SUMMARY+="\n🔸 $BASENAME: SUCCESS. Output Dir: $CURR_OUTDIR"
        fi

        # ──────────────────────────────────────────────────────────────── #
        #         [REMOVED] GENERATE FIGURES SECTION (was 06_generate_figures.py)
        # ──────────────────────────────────────────────────────────────── #
        # center_text "${YELLOW}📊 Generating Figures (Batch Mode)${NC}"
        # python Workflow/06_generate_figures.py --config "$CONFIG_FILE" --outdir "$CURR_OUTDIR"
    done

    # ──────────────────────────────────────────────────────────────── #
    #                       SUMMARY (BATCH)                           #
    # ──────────────────────────────────────────────────────────────── #
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

    # ──────────────────────────────────────────────────────────────── #
    #                GENERATE HTML REPORT (TAP-BATCH)                 #
    # ──────────────────────────────────────────────────────────────── #
    center_text "${YELLOW}📝 Generating HTML Report (Batch Mode)${NC}"
    python Workflow/05_generate_report.py --batch_outdirs "${OUTDIRS[@]}"

    # ──────────────────────────────────────────────────────────────── #
    #                       BATCH RESULTS SUMMARY                     #
    # ──────────────────────────────────────────────────────────────── #
    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${GREEN}🎯 CELLFLOW BATCH RESULTS SUMMARY${NC}"
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
    done
    echo -e "${BLUE}─────────────────────────────────────────────${NC}"
    echo -e "${GREEN}Open any report in your browser, e.g.:${NC} file://[OUTPUTDIR]/report.html"
fi
