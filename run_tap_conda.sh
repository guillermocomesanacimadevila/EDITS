#!/bin/bash

# ──────────────────────────────────────────────────────────────── #
#                   CELLFLOW PIPELINE: USER-FRIENDLY              #
#   (Auto Conda install + env setup + pipeline, interactive UI)   #
# ──────────────────────────────────────────────────────────────── #

# --- Terminal Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# --- Help Option
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo -e "${YELLOW}CELLFLOW PIPELINE${NC}"
    echo -e "Usage: bash $0"
    echo "You will be interactively prompted for input files/parameters."
    echo "Outputs go to ./runs/ and to your specified output directory."
    echo -e "After run, open your HTML report in your browser."
    exit 0
fi

trap 'echo -e "\n${RED}⚡️ Script interrupted by user. Exiting!${NC}"; exit 1' SIGINT

# --- Banner Function
banner() {
    echo -e "\n${BLUE}───────────────────────────────────────────────────────────────${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────────────${NC}"
}

# --- Interactive prompt with default, validation and explanation
prompt_val() {
    local prompt default regex explain val
    prompt="$1"
    default="$2"
    regex="$3"
    explain="$4"
    while true; do
        [ -n "$explain" ] && echo -e "${YELLOW}$explain${NC}"
        read -e -p "$(echo -e "${CYAN}$prompt${NC} [default: ${GREEN}$default${NC}]: ")" val
        val="${val:-$default}"
        if [[ -z "$regex" || "$val" =~ $regex ]]; then
            echo "$val"
            return
        else
            echo -e "${RED}Invalid input! Please try again.${NC}"
        fi
    done
}

# --- Robust file selection
select_file() {
    local prompt="$1"
    local filter="$2"
    local explain="$3"
    local files
    IFS=$'\n' read -d '' -r -a files < <(find Data -type f -iname "$filter" && printf '\0')
    if [ ${#files[@]} -eq 0 ]; then
        echo -e "${RED}❌ No $filter files found in Data/.${NC}"
        exit 1
    fi
    [ -n "$explain" ] && echo -e "${YELLOW}$explain${NC}"
    if command -v fzf &> /dev/null; then
        printf "%s\n" "${files[@]}" | fzf --prompt="$prompt > "
    else
        while true; do
            echo -e "${YELLOW}fzf not installed — fallback to numbered menu:${NC}"
            for i in "${!files[@]}"; do echo "  [$i] ${files[$i]}"; done
            echo -e "${CYAN}Type the number of your file from the list and press enter.${NC}"
            read -p "Enter number: " selection
            if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 0 ] && [ "$selection" -lt "${#files[@]}" ]; then
                echo "${files[$selection]}"
                return
            else
                echo -e "${RED}Invalid selection. Please enter a valid number from the list above.${NC}"
            fi
        done
    fi
}

# --- Robust directory selection
select_dir() {
    local prompt="$1"
    local explain="$2"
    local dirs
    IFS=$'\n' read -d '' -r -a dirs < <(find Data -type d | grep -v "^\.$" && printf '\0')
    if [ ${#dirs[@]} -eq 0 ]; then
        echo -e "${RED}❌ No directories found in Data/.${NC}"
        exit 1
    fi
    [ -n "$explain" ] && echo -e "${YELLOW}$explain${NC}"
    if command -v fzf &> /dev/null; then
        printf "%s\n" "${dirs[@]}" | fzf --prompt="$prompt > "
    else
        while true; do
            echo -e "${YELLOW}fzf not installed — fallback to numbered menu:${NC}"
            for i in "${!dirs[@]}"; do echo "  [$i] ${dirs[$i]}"; done
            echo -e "${CYAN}Type the number of your directory from the list and press enter.${NC}"
            read -p "Enter number: " selection
            if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 0 ] && [ "$selection" -lt "${#dirs[@]}" ]; then
                echo "${dirs[$selection]}"
                return
            else
                echo -e "${RED}Invalid selection. Please enter a valid number from the list above.${NC}"
            fi
        done
    fi
}

# --- Auto-install fzf (and Homebrew if needed)
install_fzf_if_missing() {
    if ! command -v fzf &> /dev/null; then
        echo -e "${YELLOW}🔎 'fzf' not found. Attempting auto-install...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS: Try Homebrew first
            if ! command -v brew &> /dev/null; then
                echo -e "${YELLOW}🍺 Homebrew is not installed. Install it? (y/n)${NC}"
                read -r REPLY
                if [[ "$REPLY" =~ ^[Yy]$ ]]; then
                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                    export PATH="/opt/homebrew/bin:$PATH"
                else
                    echo -e "${RED}Homebrew is required for fzf install on macOS. Exiting.${NC}"
                    return 1
                fi
            fi
            brew install fzf
        elif command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y fzf
        else
            echo -e "${RED}Cannot auto-install fzf: unsupported platform. Please install fzf manually!${NC}"
            return 1
        fi
        echo -e "${GREEN}✅ fzf installed!${NC}"
    fi
}
install_fzf_if_missing

# --- Conda/Miniconda auto-install/activate
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

# --- Font fix for matplotlib
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y fonts-dejavu-core fontconfig
fi
mkdir -p ~/.config/matplotlib
echo "font.family: sans-serif
font.sans-serif: DejaVu Sans
" > ~/.config/matplotlib/matplotlibrc

# --- Install TAP/tarrow in editable mode if not already
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

# --- Log versions
echo -e "${YELLOW}🔢 Environment Versions:${NC}"
echo -n "Python: "; python --version
echo -n "Conda: "; conda --version
echo -n "PyTorch: "; python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "N/A"
echo -n "Numpy: "; python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo "N/A"
echo "----------------------------"

# ──────────────────────────────────────────────────────────────── #
#                           USER INPUT                            #
# ──────────────────────────────────────────────────────────────── #
banner "🔬 CELLFLOW ML Pipeline Setup"
echo -e "${CYAN}All data file and directory selection is interactive below.${NC}"

# --- Select training movie
INPUT_TRAIN=$(select_file "🎞️  Select training movie (.tif)" "*.tif" "Choose your training .tif (raw movie, NOT mask or binary event).")
while [ ! -f "$INPUT_TRAIN" ]; do
    echo -e "${RED}❌ File not found! Try again.${NC}"
    INPUT_TRAIN=$(select_file "🎞️  Select training movie (.tif)" "*.tif")
done

# --- Choose validation mode
while true; do
    echo -e "${CYAN}Do you want to:${NC}\n  0: Validate on a single movie\n  1: Validate on all .tif files in a directory (no classifier, TAP metrics only!)"
    read -p "Select option (0 or 1) [default: 0]: " VAL_BATCH
    VAL_BATCH="${VAL_BATCH:-0}"
    [[ "$VAL_BATCH" == "0" || "$VAL_BATCH" == "1" ]] && break
    echo -e "${RED}Please enter 0 or 1.${NC}"
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ "$VAL_BATCH" == "0" ]; then
    # --- Single validation
    INPUT_VAL=$(select_file "🧪  Select validation movie (.tif)" "*.tif" "Select the validation .tif file (should differ from training movie).")
    while [ ! -f "$INPUT_VAL" ]; do
        echo -e "${RED}❌ File not found! Try again.${NC}"
        INPUT_VAL=$(select_file "🧪  Select validation movie (.tif)" "*.tif")
    done

    INPUT_MASK=$(select_file "🎭  Select annotated mask (.tif)" "*mask*.tif" "Select the corresponding annotated mask (.tif) for validation movie. You can also pick a file with 'annotated' in its name.")
    while [ ! -f "$INPUT_MASK" ]; do
        echo -e "${RED}❌ File not found! Try again.${NC}"
        INPUT_MASK=$(select_file "🎭  Select annotated mask (.tif)" "*mask*.tif")
    done

    CROP_SIZE=$(prompt_val "📐 Enter crop size (integer, e.g. 48)" "48" '^[0-9]+$' "Typical: 48-128, must be an integer.")
    PIXEL_RES=$(prompt_val "🔬 Enter pixel resolution (float, e.g. 0.65)" "0.65" '^[0-9]+(\.[0-9]+)?$' "Typical: 0.5-1.0 microns/pixel.")
    EPOCHS=$(prompt_val "🔁 Number of training epochs" "25" '^[0-9]+$' "Number of training epochs (integer, e.g. 25).")
    OUTDIR=$(prompt_val "📂 Output directory path" "Data/toy_data" '' "Where outputs/reports will be saved.")
    SEED=$(prompt_val "🎲 Random seed (integer)" "42" '^[0-9]+$' "Random seed for reproducibility.")
    while true; do
        BACKBONE=$(prompt_val "🧠 Backbone (unet, spectformer-xs)" "unet" '' "Type 'unet' or 'spectformer-xs'.")
        [[ "$BACKBONE" == "unet" || "$BACKBONE" == "spectformer-xs" ]] && break
        echo -e "${RED}Unsupported backbone! Must be 'unet' or 'spectformer-xs'.${NC}"
    done

    # --- Review selections before continuing
    banner "⚡️ Review your selections!"
    echo -e "${YELLOW}Training movie :${NC} $INPUT_TRAIN"
    echo -e "${YELLOW}Validation movie:${NC} $INPUT_VAL"
    echo -e "${YELLOW}Mask file      :${NC} $INPUT_MASK"
    echo -e "${YELLOW}Crop size      :${NC} $CROP_SIZE"
    echo -e "${YELLOW}Pixel res      :${NC} $PIXEL_RES"
    echo -e "${YELLOW}Epochs         :${NC} $EPOCHS"
    echo -e "${YELLOW}Output dir     :${NC} $OUTDIR"
    echo -e "${YELLOW}Seed           :${NC} $SEED"
    echo -e "${YELLOW}Backbone       :${NC} $BACKBONE"
    echo
    read -p "$(echo -e "${CYAN}Proceed with these settings? (y/n)${NC} ")" CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Cancelled by user. Restarting...${NC}"
        exec "$0"
    fi

    # --- Run pipeline
    MODEL_ID="$(basename "$INPUT_TRAIN" .tif)_${BACKBONE}_$TIMESTAMP"
    MODEL_RUN_DIR="runs/${MODEL_ID}"
    mkdir -p "$MODEL_RUN_DIR"
    OUTDIR="${OUTDIR%/}_${MODEL_ID}"
    mkdir -p "$OUTDIR"
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

    banner "📝 Configuration saved to $CONFIG_FILE"
    LOGFILE="$OUTDIR/pipeline_log.txt"
    exec > >(tee -i "$LOGFILE")
    exec 2>&1
    START_TIME=$(date +%s)

    banner "🚀 Training Model (Fine-tuning)"
    python Workflow/01_fine-tune.py --config "$CONFIG_FILE" || { echo -e "${RED}❌ Fine-tuning failed!${NC}"; exit 1; }

    banner "🚀 Data Preparation"
    python Workflow/02_data_prep.py \
        --input_frame "$INPUT_VAL" \
        --input_mask "$INPUT_MASK" \
        --data_save_dir "$OUTDIR" \
        --size "$CROP_SIZE" \
        --pixel_area_threshold 0 \
        --binarize \
        --data_seed "$SEED" \
        || { echo -e "${RED}❌ Data prep failed!${NC}"; exit 1; }

    TAP_MODEL_DIR="${MODEL_RUN_DIR}/${MODEL_ID}_backbone_${BACKBONE}"
    echo "TAP model folder: $TAP_MODEL_DIR"
    ls -lh "$TAP_MODEL_DIR"

    banner "🚀 Event Classification"
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

    banner "🚀 Examining Mistaken Predictions"
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

    banner "📊 Generating Publication-Ready Figures"
    python Workflow/06_generate_figures.py --config "$CONFIG_FILE" --outdir "$OUTDIR"

    banner "🎉 CELLFLOW Pipeline Complete!"
    echo -e "${YELLOW}Model ID      :${NC} $MODEL_ID"
    echo -e "${YELLOW}Model Dir     :${NC} $MODEL_RUN_DIR"
    echo -e "${YELLOW}Output Dir    :${NC} $OUTDIR"
    echo -e "${YELLOW}Crop Size     :${NC} $CROP_SIZE"
    echo -e "${YELLOW}Epochs        :${NC} $EPOCHS"
    echo -e "${YELLOW}Pixel Res     :${NC} $PIXEL_RES"
    echo -e "${YELLOW}Backbone      :${NC} $BACKBONE"
    echo -e "${YELLOW}Mask File     :${NC} $INPUT_MASK"
    echo -e "${YELLOW}Config File   :${NC} $CONFIG_FILE"
    echo -e "${YELLOW}Log File      :${NC} $LOGFILE"
    echo -e "${YELLOW}Runtime       :${NC} $((RUNTIME / 60)) min $((RUNTIME % 60)) sec"

    banner "📝 Generating HTML Report"
    python Workflow/05_generate_report.py --config "$CONFIG_FILE" --outdir "$OUTDIR"

    # Try to open the HTML report automatically, if possible
    if [ -f "$OUTDIR/report.html" ]; then
        if command -v xdg-open &> /dev/null; then xdg-open "$OUTDIR/report.html" &>/dev/null; fi
        if command -v open &> /dev/null; then open "$OUTDIR/report.html" &>/dev/null; fi
        if command -v start &> /dev/null; then start "$OUTDIR/report.html" &>/dev/null; fi
        echo -e "${GREEN}Open your report in your browser: file://$OUTDIR/report.html${NC}"
    else
        echo -e "${RED}HTML report was not generated. Please check logs.${NC}"
    fi

else
    # --- Batch validation mode
    VAL_DIR=$(select_dir "📂  Select directory with validation .tif files" "Choose the directory containing your validation .tif files.")
    CROP_SIZE=$(prompt_val "📐 Enter crop size (integer, e.g. 48)" "48" '^[0-9]+$')
    PIXEL_RES=$(prompt_val "🔬 Enter pixel resolution (float, e.g. 0.65)" "0.65" '^[0-9]+(\.[0-9]+)?$')
    OUTDIR=$(prompt_val "📂 Output directory path" "Data/toy_data" '')
    SEED=$(prompt_val "🎲 Random seed (integer)" "42" '^[0-9]+$')
    while true; do
        BACKBONE=$(prompt_val "🧠 Backbone (unet, spectformer-xs)" "unet" '')
        [[ "$BACKBONE" == "unet" || "$BACKBONE" == "spectformer-xs" ]] && break
        echo -e "${RED}Unsupported backbone! Must be 'unet' or 'spectformer-xs'.${NC}"
    done

    banner "⚡️ Review your selections!"
    echo -e "${YELLOW}Training movie :${NC} $INPUT_TRAIN"
    echo -e "${YELLOW}Validation dir :${NC} $VAL_DIR"
    echo -e "${YELLOW}Crop size      :${NC} $CROP_SIZE"
    echo -e "${YELLOW}Pixel res      :${NC} $PIXEL_RES"
    echo -e "${YELLOW}Output dir     :${NC} $OUTDIR"
    echo -e "${YELLOW}Seed           :${NC} $SEED"
    echo -e "${YELLOW}Backbone       :${NC} $BACKBONE"
    echo
    read -p "$(echo -e "${CYAN}Proceed with these settings? (y/n)${NC} ")" CONFIRM
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Cancelled by user. Restarting...${NC}"
        exec "$0"
    fi

    VAL_FILES=()
    while IFS= read -r f; do VAL_FILES+=("$f"); done < <(find "$VAL_DIR" -maxdepth 1 -type f -iname "*.tif" | sort)
    if [ ${#VAL_FILES[@]} -eq 0 ]; then
        echo -e "${RED}❌ No .tif files found in $VAL_DIR!${NC}"
        exit 1
    fi

    banner "📂 Found ${#VAL_FILES[@]} validation movies"
    for v in "${VAL_FILES[@]}"; do echo "   - $v"; done

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

        banner "🚀 TAP-only eval for $BASENAME"
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

        banner "📊 Generating Figures (Batch Mode)"
        python Workflow/06_generate_figures.py --config "$CONFIG_FILE" --outdir "$CURR_OUTDIR"
    done

    banner "🎉 Batch TAP Validation Complete!"
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"
    echo "🔹 Training movie: $INPUT_TRAIN"
    echo "🔹 Crop Size     : $CROP_SIZE"
    echo "🔹 Pixel Res     : $PIXEL_RES"
    echo "🔹 Backbone      : $BACKBONE"
    echo "🔹 Random Seed   : $SEED"
    echo "🔹 Validation dir: $VAL_DIR"
    echo -e "🔹 Results: $RUN_SUMMARY"
    echo -e "${GREEN}───────────────────────────────────────────────────────────────${NC}"

    banner "📝 Generating HTML Report (Batch Mode)"
    python Workflow/05_generate_report.py --batch_outdirs "${OUTDIRS[@]}"
    for d in "${OUTDIRS[@]}"; do
        if [ -f "$d/report.html" ]; then
            if command -v xdg-open &> /dev/null; then xdg-open "$d/report.html" &>/dev/null; fi
            if command -v open &> /dev/null; then open "$d/report.html" &>/dev/null; fi
            if command -v start &> /dev/null; then start "$d/report.html" &>/dev/null; fi
        fi
    done
    echo -e "${GREEN}Open any report in your browser, e.g.: file://[OUTPUTDIR]/report.html${NC}"
fi
