# ──────────────────────────────────────────────────────────────── #
#                          USER INPUT                             #
# ──────────────────────────────────────────────────────────────── #
center_text "🔬 CELLFLOW ML Pipeline Setup"
echo "ℹ️  Please provide paths relative to the project root (e.g. Data/toy_data/toy_movie.tif)"
echo "   Do NOT use absolute paths like /Users/yourname/Desktop/..."

read -p "$(center_text '📥 Path to training movie (.tif, e.g. Data/toy_data/toy_movie.tif):')" INPUT_TRAIN

center_text "🧪 Validate on just 1 movie or on a whole directory?"
echo "   0: Single validation movie"
echo "   1: Validate on every .tif in a folder (no classifier, TAP metrics only!)"
read -p "Select option (0 or 1): " VAL_BATCH

if [[ "$VAL_BATCH" != "0" && "$VAL_BATCH" != "1" ]]; then
    echo "❌ Invalid input. Please enter 0 or 1."
    exit 1
fi

if [ "$VAL_BATCH" == "0" ]; then
    # ───────────── Single Validation Movie Mode ───────────── #
    read -p "$(center_text '🧪 Path to validation movie (.tif, e.g. Data/toy_data/toy_movie.tif):')" INPUT_VAL
    read -p "$(center_text '🎭 Path to annotated mask (.tif, e.g. Data/toy_data/toy_mask.tif):')" INPUT_MASK
    read -p "$(center_text '📐 Crop size (e.g., 48):')" CROP_SIZE
    read -p "$(center_text '🔬 Pixel resolution (e.g., 0.65):')" PIXEL_RES
    read -p "$(center_text '🔁 Number of training epochs:')" EPOCHS
    read -p "$(center_text '📛 Model ID (e.g., cellflow_2025):')" MODEL_ID
    read -p "$(center_text '📂 Output directory path (e.g., Data/toy_data):')" OUTDIR
    read -p "$(center_text '🎲 Random seed:')" SEED
    read -p "$(center_text '🧠 Backbone (unet, spectformer-xs):')" BACKBONE

    # ───────────── Input validation and config creation, same as your script ───────────── #
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

    MODEL_RUN_DIR="runs/${MODEL_ID}_backbone_${BACKBONE}_$TIMESTAMP"
    mkdir -p "$MODEL_RUN_DIR"
    echo "📂 Model run folder will be: $MODEL_RUN_DIR"

    OUTDIR="${OUTDIR%/}_${MODEL_ID}_$TIMESTAMP"
    mkdir -p "$OUTDIR"
    echo "📂 Output folder will be: $OUTDIR"

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

    center_text "📝 Configuration saved to $CONFIG_FILE"

    # Remove submodel folder just before Nextflow
    SUBMODEL_DIR="$MODEL_RUN_DIR/${MODEL_ID}_backbone_${BACKBONE}"
    if [ -d "$SUBMODEL_DIR" ]; then
        echo "⚠️  Removing previous submodel folder: $SUBMODEL_DIR"
        rm -rf "$SUBMODEL_DIR"
    fi

    # Logging setup
    LOGFILE="$OUTDIR/pipeline_log.txt"
    exec > >(tee -i "$LOGFILE")
    exec 2>&1

    # Docker check (same as your script)
    if ! docker info >/dev/null 2>&1; then
        echo "🐳 Docker is not running. Attempting to start Docker Desktop..."
        open -a Docker
        WAIT_COUNT=0
        until docker info >/dev/null 2>&1; do
            sleep 2
            ((WAIT_COUNT+=2))
            if [ $WAIT_COUNT -ge 60 ]; then
                echo "❌ Docker did not start within 60 seconds. Please check Docker Desktop manually."
                exit 1
            fi
            echo "⏳ Waiting for Docker to start... ($WAIT_COUNT/60 sec)"
        done
        echo "✅ Docker is now running!"
    fi

    # Run Nextflow (as usual)
    center_text "🚀 Running CELLFLOW with Nextflow"
    START_TIME=$(date +%s)
    nextflow run main.nf -params-file "$CONFIG_FILE" --model_run_dir "$MODEL_RUN_DIR"
    NFX_EXIT=$?
    END_TIME=$(date +%s)
    RUNTIME=$((END_TIME - START_TIME))

    # ──────────────────────────────────────────────────────────────── #
    #                           SUMMARY                               #
    # ──────────────────────────────────────────────────────────────── #
    if [ $NFX_EXIT -ne 0 ]; then
        echo -e "\n❌ Pipeline failed. Check the log file: $LOGFILE"
        exit 1
    fi

    center_text "🎉 CELLFLOW Pipeline Complete!"
    echo "───────────────────────────────────────────────────────────────"
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
    echo "───────────────────────────────────────────────────────────────"
    echo "🙏 If you use CELLFLOW in your research, please cite the corresponding paper."

else
    # ───────────── Batch Validation Mode ───────────── #
    read -p "$(center_text '📂 Directory containing validation .tif files (e.g. Data/Validation):')" VAL_DIR
    read -p "$(center_text '📐 Crop size (e.g., 48):')" CROP_SIZE
    read -p "$(center_text '🔬 Pixel resolution (e.g., 0.65):')" PIXEL_RES
    read -p "$(center_text '📛 Model ID (e.g., cellflow_2025):')" MODEL_ID
    read -p "$(center_text '📂 Output directory path (e.g., Data/toy_data):')" OUTDIR
    read -p "$(center_text '🎲 Random seed:')" SEED
    read -p "$(center_text '🧠 Backbone (unet, spectformer-xs):')" BACKBONE

    re_int='^[0-9]+$'
    re_float='^[0-9]+(\.[0-9]+)?$'
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)

    [ ! -f "$INPUT_TRAIN" ] && echo "❌ Training movie not found at '$INPUT_TRAIN'" && exit 1
    [ ! -d "$VAL_DIR" ] && echo "❌ Validation directory not found at '$VAL_DIR'" && exit 1
    [[ ! "$CROP_SIZE" =~ $re_int ]] && echo "❌ Crop size must be an integer." && exit 1
    [[ ! "$SEED" =~ $re_int ]] && echo "❌ Seed must be an integer." && exit 1
    [[ ! "$PIXEL_RES" =~ $re_float ]] && echo "❌ Pixel resolution must be a float." && exit 1
    [[ "$BACKBONE" != "unet" && "$BACKBONE" != "spectformer-xs" ]] && echo "❌ Unsupported backbone: $BACKBONE" && exit 1

    # Find all .tif files in validation dir
    mapfile -t VAL_FILES < <(find "$VAL_DIR" -maxdepth 1 -type f -iname "*.tif" | sort)
    if [ "${#VAL_FILES[@]}" -eq 0 ]; then
        echo "❌ No .tif files found in $VAL_DIR!"
        exit 1
    fi

    echo "📂 Found ${#VAL_FILES[@]} validation movies:"
    for v in "${VAL_FILES[@]}"; do
        echo "   - $v"
    done

    # MAIN LOOP
    RUN_SUMMARY=""
    for VAL_MOVIE in "${VAL_FILES[@]}"; do
        BASENAME=$(basename "$VAL_MOVIE" .tif)
        CURR_OUTDIR="${OUTDIR%/}_${MODEL_ID}_${BASENAME}_$TIMESTAMP"
        mkdir -p "$CURR_OUTDIR"
        MODEL_RUN_DIR="runs/${MODEL_ID}_backbone_${BACKBONE}_${BASENAME}_$TIMESTAMP"
        mkdir -p "$MODEL_RUN_DIR"
        CONFIG_FILE="$CURR_OUTDIR/run_config.yaml"

        # TAP ONLY: Don't require mask or run classifier
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

        center_text "🚀 TAP-only eval for $BASENAME"
        LOGFILE="$CURR_OUTDIR/pipeline_log.txt"
        exec > >(tee -i "$LOGFILE")
        exec 2>&1

        # Remove submodel folder just before Nextflow
        SUBMODEL_DIR="$MODEL_RUN_DIR/${MODEL_ID}_backbone_${BACKBONE}"
        if [ -d "$SUBMODEL_DIR" ]; then
            echo "⚠️  Removing previous submodel folder: $SUBMODEL_DIR"
            rm -rf "$SUBMODEL_DIR"
        fi

        # Docker check (as before)
        if ! docker info >/dev/null 2>&1; then
            echo "🐳 Docker is not running. Attempting to start Docker Desktop..."
            open -a Docker
            WAIT_COUNT=0
            until docker info >/dev/null 2>&1; do
                sleep 2
                ((WAIT_COUNT+=2))
                if [ $WAIT_COUNT -ge 60 ]; then
                    echo "❌ Docker did not start within 60 seconds. Please check Docker Desktop manually."
                    exit 1
                fi
                echo "⏳ Waiting for Docker to start... ($WAIT_COUNT/60 sec)"
            done
            echo "✅ Docker is now running!"
        fi

        # Run Nextflow (TAP mode only)
        START_TIME=$(date +%s)
        nextflow run main.nf -params-file "$CONFIG_FILE" --model_run_dir "$MODEL_RUN_DIR" --tap_only true
        NFX_EXIT=$?
        END_TIME=$(date +%s)
        RUNTIME=$((END_TIME - START_TIME))

        if [ $NFX_EXIT -ne 0 ]; then
            echo -e "\n❌ TAP eval failed for $BASENAME. Check the log file: $LOGFILE"
            RUN_SUMMARY+="\n🔸 $BASENAME: FAILED! See $LOGFILE"
        else
            echo -e "\n✅ TAP eval complete for $BASENAME."
            RUN_SUMMARY+="\n🔸 $BASENAME: SUCCESS. Output Dir: $CURR_OUTDIR"
        fi
    done

    # ──────────────────────────────────────────────────────────────── #
    #                           SUMMARY (BATCH)                       #
    # ──────────────────────────────────────────────────────────────── #
    center_text "🎉 Batch TAP Validation Complete!"
    echo "───────────────────────────────────────────────────────────────"
    echo "🔹 Model ID      : $MODEL_ID"
    echo "🔹 Training movie: $INPUT_TRAIN"
    echo "🔹 Crop Size     : $CROP_SIZE"
    echo "🔹 Pixel Res     : $PIXEL_RES"
    echo "🔹 Backbone      : $BACKBONE"
    echo "🔹 Random Seed   : $SEED"
    echo "🔹 Validation dir: $VAL_DIR"
    echo -e "🔹 Results: $RUN_SUMMARY"
    echo "───────────────────────────────────────────────────────────────"
    echo "🙏 If you use CELLFLOW in your research, please cite the corresponding paper."
fi
