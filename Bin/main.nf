#!/usr/bin/env nextflow

nextflow.enable.dsl=2

// --- PARAMETERS ---
params.input_train   = null
params.input_val     = null
params.input_mask    = null
params.size          = 48
params.epochs        = 10
params.seed          = 42
params.name          = 'cellflow_run'
params.backbone      = 'unet'
params.projhead      = 'minimal_batchnorm'
params.classhead     = 'minimal'
params.batchsize     = 64
params.config_yaml   = null
params.model_run_dir = params.model_run_dir ?: 'runs'    // <--- ADDED THIS LINE

// --- CONTAINER PATH HELPERS ---
def train_path_container = params.input_train ? params.input_train[0].replaceFirst('Data/', '/data/') : ''
def val_path_container   = params.input_val   ? params.input_val[0].replaceFirst('Data/', '/data/')   : ''
def mask_path_container  = params.input_mask  ? params.input_mask[0].replaceFirst('Data/', '/data/')  : ''
def config_path_container = params.config_yaml ? params.config_yaml.replaceFirst('Data/', '/data/')   : ''

workflow {

    // Step 0: Prepare data
    prep = data_prep()

    // Step 1: Train TAP model
    tap  = train_tap_model()

    // Step 2: Train classifier head
    cls  = train_cls_head(tap.tap_model_dir, prep.test_data)

    // Step 3: Analyze predictions
    probe = probe_model(cls.cls_model, prep.test_data, tap.tap_model_dir)
}

// --- DATA PREP ---
process data_prep {

    tag "🧼 Data preprocessing"

    output:
    path 'dataset_output/test_data_crops_flat.pth', emit: test_data

    container 'tap_pipeline:latest'

    script:
    """
    CUDA_VISIBLE_DEVICES=0 python /app/Workflow/02_data_prep.py \
        --input_frame '${train_path_container}' \
        --input_mask '${mask_path_container}' \
        --data_save_dir dataset_output \
        --size ${params.size} \
        --ndim 2 \
        --frames 2 \
        --pixel_area_threshold -1 \
        --crops_per_image 1 \
        --subsample 1 \
        --binarize

    # Move output for Nextflow tracking
    if [ -f dataset_output/preprocessed_image_crops.pth ]; then
        mv dataset_output/preprocessed_image_crops.pth dataset_output/test_data_crops_flat.pth
    fi
    """
}

// --- TRAIN TAP MODEL ---
process train_tap_model {

    tag "🚀 Train TAP model"

    output:
    path 'runs', emit: tap_model_dir

    container 'tap_pipeline:latest'

    script:
    """
    # Clean the whole model run dir if exists (not just the subdir)
    if [ -d '${params.model_run_dir}' ]; then
        echo "⚠️  Removing stale model run folder: ${params.model_run_dir}"
        rm -rf '${params.model_run_dir}'
    fi

    # Also check and remove submodel dir, just in case (belt & suspenders)
    SUBMODEL_DIR='${params.model_run_dir}/${params.name}_backbone_${params.backbone}'
    if [ -d "\$SUBMODEL_DIR" ]; then
        echo "⚠️  Removing stale submodel folder: \$SUBMODEL_DIR"
        rm -rf "\$SUBMODEL_DIR"
    fi

    CUDA_VISIBLE_DEVICES=0 python /app/Workflow/01_fine-tune.py \
        -c '${config_path_container}' \
        --input_train '${train_path_container}' \
        --input_val '${val_path_container}' \
        --outdir '${params.model_run_dir}' \
        --gpu 0 \
        --epochs ${params.epochs} \
        --name ${params.name} \
        --size ${params.size} \
        --backbone ${params.backbone} \
        --seed ${params.seed} \
        --projhead ${params.projhead} \
        --classhead ${params.classhead} \
        --batchsize ${params.batchsize}
    """
}

// --- TRAIN CLASSIFIER HEAD ---
process train_cls_head {

    tag "🧠 Train classifier head"

    input:
    path tap_model_dir
    path test_data

    output:
    path 'cls_model.pth', emit: cls_model
    path 'dataset_output/test_data_crops_flat.pth', emit: test_data

    container 'tap_pipeline:latest'

    script:
    """
    CUDA_VISIBLE_DEVICES=0 python /app/Workflow/03_event_classification.py \
        --input_frame '${train_path_container}' \
        --input_mask '${mask_path_container}' \
        --data_save_dir ./ \
        --dataset_save_dir dataset_output \
        --model_save_dir ./ \
        --TAP_model_load_path \${tap_model_dir} \
        --cls_head_arch ${params.classhead} \
        --model_id ${params.name} \
        --crops_per_image 1 \
        --balanced_sample_size 2500 \
        --size ${params.size} \
        --batchsize ${params.batchsize} \
        --training_epochs ${params.epochs} \
        --model_seed ${params.seed} \
        --data_seed ${params.seed} \
        --num_runs 1 \
        --TAP_init loaded
    """
}

// --- PROBE/ANALYZE MODEL ---
process probe_model {

    tag "🔍 Analyze predictions"

    input:
    path cls_model
    path test_data
    path tap_model_dir

    output:
    path 'mistake_analysis'

    container 'tap_pipeline:latest'

    script:
    """
    CUDA_VISIBLE_DEVICES=0 python /app/Workflow/04_examine_mistaken_predictions.py \
        --masks_path '${mask_path_container}' \
        --mistake_pred_dir mistake_analysis \
        --TAP_model_load_path \${tap_model_dir} \
        --test_data_load_path \${test_data} \
        --combined_model_load_dir ./ \
        --model_id ${params.name} \
        --cls_head_arch ${params.classhead} \
        --patch_size ${params.size} \
        --is_true_positive \
        --is_true_negative \
        --num_egs_to_show 10 \
        --save_data
    """
}
