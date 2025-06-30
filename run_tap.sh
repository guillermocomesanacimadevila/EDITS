#!/bin/bash

# ──────────────────────────────────────────────────────────────── #
#                        CELLFLOW PIPELINE                        #
# ──────────────────────────────────────────────────────────────── #

center_text() {
    local width=70
    local text="$1"
    printf "\n%*s\n\n" $(( (${#text} + width) / 2 )) "$text"
}

# ──────────────────────────────────────────────────────────────── #
#                AUTO-START DOCKER IF NOT RUNNING                 #
# ──────────────────────────────────────────────────────────────── #

if ! docker info > /dev/null 2>&1; then
    echo "⏳ Docker is not running."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🚀 Attempting to start Docker Desktop on Mac..."
        open -a Docker
        echo "⌛ Waiting for Docker to launch..."
        while ! docker info > /dev/null 2>&1; do sleep 2; done
        echo "✅ Docker is running."
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "🚀 Attempting to start Docker service on Linux..."
        sudo systemctl start docker
        sleep 5
        if ! docker info > /dev/null 2>&1; then
            echo "❌ Docker is still not running. Please start it manually."
            exit 1
        fi
        echo "✅ Docker is running."
    else
        echo "⚠️  Please start Docker Desktop manually!"
        read -p "Press [Enter] when Docker is running..."
        if ! docker info > /dev/null 2>&1; then
            echo "❌ Docker is still not running. Exiting."
            exit 1
        fi
    fi
fi

# ──────────────────────────────────────────────────────────────── #
#                CHECK & OFFER TO INSTALL NEXTFLOW                #
# ──────────────────────────────────────────────────────────────── #

if ! command -v nextflow &>/dev/null; then
    echo "⚠️  Nextflow is not installed."
    read -p "Would you like to auto-install Nextflow? (y/n): " INSTALL_NEXTFLOW
    if [[ "$INSTALL_NEXTFLOW" == "y" || "$INSTALL_NEXTFLOW" == "Y" ]]; then
        curl -s https://get.nextflow.io | bash
        sudo mv nextflow /usr/local/bin/
        sudo chmod +x /usr/local/bin/nextflow
        echo "✅ Nextflow installed!"
    else
        echo "❌ Nextflow is required. Exiting."
        exit 1
    fi
fi

# ──────────────────────────────────────────────────────────────── #
#                          USER INPUT                             #
# ──────────────────────────────────────────────────────────────── #

center_text "🔬 CELLFLOW ML Pipeline Setup"
echo "ℹ️  Please provide paths relative to the project root (e.g. Data/toy_data/toy_movie.tif)"
echo "   Do NOT use absolute paths like /Users/yourname/Desktop/..."

read -p "$(center_text '📥 Path to training movie (.tif, e.g. Data/toy_data/toy_movie.tif):')" INPUT_TRAIN
read -p "$(center_text '🧪 Path to validation movie (.tif, e.g. Data/toy_data/toy_movie.tif):')" INPUT_VAL
read -p "$(center_text '🎭 Path to annotated mask (.tif, e.g. Data/toy_data/toy_mask.tif):')" INPUT_MASK
read -p "$(center_text '📐 Crop size (e.g., 48):')" CROP_SIZE
read -p "$(center_text '🔬 Pixel resolution (e.g., 0.65):')" PIXEL_RES
read -p "$(center_text '🔁 Number of training epochs:')" EPOCHS
read -p "$(center_text '📛 Model ID (e.g., cellflow_2025):')" MODEL_ID
read -p "$(center_text '📂 Output directory path (e.g., Data/toy_data):')" OUTDIR
read -p "$(center_text '🎲 Random seed:')" SEED
read -p "$(center_text '🧠 Backbone (unet, spectformer-xs):')" BACKBONE

# ──────────────────────────────────────────────────────────────── #
#                        PATH VALIDATION                          #
# ──────────────────────────────────────────────────────────────── #

re_int='^[0-9]+$'
re_float='^[0-9]+(\.[0-9]+)?$'
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

[ ! -f "$INPUT_TRAIN" ] && echo "❌ Training movie not found at '$INPUT_TRAIN'" && exit 1
[ ! -f "$INPUT_VAL" ] && echo "❌ Validation movie not found at '$INPUT_VAL'" && exit 1
[ ! -f "$INPUT_MASK" ] && echo "❌ Mask not found at '$INPUT_MASK'" && exit 1
[[ ! "$CROP_SIZE" =~ $re_int ]] && echo "❌ Crop size must be an integer." && exit 1
[[ ! "$EPOCHS" =~ $re_int ]] && echo "❌ Epochs must be an integer." && exit 1
[[ ! "$SEED" =~ $re_int ]] && echo "❌ Seed must be an integer." && exit 1
[[ ! "$PIXEL_RES" =~ $re_float ]] && echo "❌ Pixel resolution must be a float." && exit 1
[[ "$BACKBONE" != "unet" && "$BACKBONE" != "spectformer-xs" ]] && echo "❌ Unsupported backbone: $BACKBONE" && exit 1

OUTDIR="${OUTDIR%/}_${MODEL_ID}_$TIMESTAMP"
mkdir -p "$OUTDIR"

# ──────────────────────────────────────────────────────────────── #
#                  CONTAINER PATHS FOR DOCKER                     #
# ──────────────────────────────────────────────────────────────── #
# Mounts the 'Data' folder to '/data' inside the container.

CONTAINER_TRAIN="/data/${INPUT_TRAIN#Data/}"
CONTAINER_VAL="/data/${INPUT_VAL#Data/}"
CONTAINER_MASK="/data/${INPUT_MASK#Data/}"

# Show mapping info to user:
echo "------------------------------------------"
echo "Using host training file:   $INPUT_TRAIN"
echo "Using Docker training file: $CONTAINER_TRAIN"
echo "------------------------------------------"

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
  - "$CONTAINER_TRAIN"
input_val:
  - "$CONTAINER_VAL"
input_mask:
  - "$CONTAINER_MASK"
split_train:
  - [0.0, 1.0]
split_val:
  - [0.0, 1.0]
outdir: "$OUTDIR"
gpu: "0"
seed: $SEED
pixel_resolution: $PIXEL_RES
tensorboard: true
write_final_cams: false
binarize: false
config_yaml: "$CONFIG_FILE"
EOL

center_text "📝 Configuration saved to $CONFIG_FILE"

# ──────────────────────────────────────────────────────────────── #
#                        OPTIONAL DOCKER BUILD                    #
# ──────────────────────────────────────────────────────────────── #

read -p "$(center_text '🐳 Build Docker image? (y/n):')" BUILD_DOCKER
if [[ "$BUILD_DOCKER" == "y" || "$BUILD_DOCKER" == "Y" ]]; then
    docker build -t tap_pipeline:latest .
    center_text "✅ Docker image built: tap_pipeline:latest"
fi

# ──────────────────────────────────────────────────────────────── #
#                          RESUME OPTION                          #
# ──────────────────────────────────────────────────────────────── #

read -p "$(center_text '⏯️ Resume previous run if exists? (y/n):')" RESUME_FLAG
RESUME_OPTION=""
[ "$RESUME_FLAG" == "y" ] && RESUME_OPTION="-resume"

# ──────────────────────────────────────────────────────────────── #
#                         LOGGING SETUP                           #
# ──────────────────────────────────────────────────────────────── #

LOGFILE="$OUTDIR/pipeline_log.txt"
exec > >(tee -i "$LOGFILE")
exec 2>&1

# ──────────────────────────────────────────────────────────────── #
#                        NEXTFLOW EXECUTION                       #
# ──────────────────────────────────────────────────────────────── #

center_text "🚀 Running CELLFLOW with Nextflow"

nextflow run main.nf \
  -with-docker tap_pipeline:latest \
  -params-file "$CONFIG_FILE" \
  $RESUME_OPTION

# ──────────────────────────────────────────────────────────────── #
#                           SUMMARY                               #
# ──────────────────────────────────────────────────────────────── #

center_text "🎉 CELLFLOW Pipeline Complete!"
echo "🔹 Model ID      : $MODEL_ID"
echo "🔹 Output Dir    : $OUTDIR"
echo "🔹 Crop Size     : $CROP_SIZE"
echo "🔹 Epochs        : $EPOCHS"
echo "🔹 Pixel Res     : $PIXEL_RES"
echo "🔹 Backbone      : $BACKBONE"
echo "🔹 Mask File     : $CONTAINER_MASK"
echo "🔹 Config File   : $CONFIG_FILE"
echo "🔹 Log File      : $LOGFILE"
