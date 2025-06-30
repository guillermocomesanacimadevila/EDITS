#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.config_yaml = null

workflow {

    // Step 0: Prepare data
    prep_output = data_prep()

    // Step 1: Train TAP model
    tap_output = train_tap_model()

    // Step 2: Train classifier head
    cls_output = train_cls_head(tap_output.tap_model_dir, prep_output.test_data)

    // Step 3: Analyze model predictions
    probe_model(cls_output.cls_model, cls_output.test_data, tap_output.tap_model_dir)
}

process data_prep {

    tag "🧼 Data preprocessing"

    output:
    path 'dataset_output/test_data_crops_flat.pth', emit: test_data

    container 'tap_pipeline:latest'

    script:
    """
    CUDA_VISIBLE_DEVICES=0 python /app/Workflow/02_data_prep.py \
        --input_frame '${params.input_train[0]}' \
        --input_mask '${params.input_mask[0]}' \
        --data_save_dir dataset_output \
        --size ${params.size} \
        --ndim 2 \
        --frames 2 \
        --pixel_area_threshold -1 \
        --crops_per_image 1 \
        --subsample 1 \
        --binarize
    """
}

process train_tap_model {

    tag "🚀 Train TAP model"

    output:
    path 'runs', emit: tap_model_dir

    container 'tap_pipeline:latest'

    script:
    """
    CUDA_VISIBLE_DEVICES=0 python /app/Workflow/01_fine-tune.py \
        -c '${params.config_yaml}' \
        --input_train '${params.input_train[0]}' \
        --input_val '${params.input_val[0]}' \
        --outdir runs \
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
        --input_frame '${params.input_train[0]}' \
        --input_mask '${params.input_mask[0]}' \
        --data_save_dir ./ \
        --dataset_save_dir dataset_output \
        --model_save_dir ./ \
        --TAP_model_load_path ${tap_model_dir} \
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
        --masks_path '${params.input_mask[0]}' \
        --mistake_pred_dir mistake_analysis \
        --TAP_model_load_path ${tap_model_dir} \
        --test_data_load_path ${test_data} \
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
