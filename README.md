# 📸 Synergy Project – Live Cell Microscopy

Temporal feature learning for event classification in live-cell imaging.

Authors:
Cangxiong Chen, Guillermo Comesaña Cimadevila, Vinay P. Namboodiri, Julia E. Sero

### Run

```bash
git clone https://github.com/guillermocomesanacimadevila/Synergy_project.git
cd Synergy_project
```

```bash
cd ~/Synergy_project && bash run_tap_conda.sh
```

### Test configuration example
--model_id resnet_head_2024-09-10-2359 --crops_per_image 1000 
--balanced_sample_size 500000 --size 48 --batchsize 108 
--training_epochs 30 --model_seed 20 --data_seed 25 --num_runs 10

### Example usage
#### TAP Pretraining
In this step, we pretrain a time arrow prediction model to get dense features for the downstream task.
```
CUDA_VISIBLE_DEVICES=1 python fine-tune.py \ 
 -c synergy_config.yaml \ # attaching configuration file 
 --input_train /path/021221_C16-1_8bit_PFFC-BrightAdj.tif \ # path to movie to train the model
 --input_val /path/021221_C16-1_8bit_PFFC-BrightAdj.tif \ # path to movie to valid the model (can be the same)
 --outdir /path \ # path to store the trained model as a PyTorch pth file
 --gpu 0 \ # (legacy) specifying GPU device if applicable 

```

#### Example of a configuration file (.yaml) for TAP pretraining
```
name: synergy  # customisable experiment ID
epochs: 10  # epochs for TAP pretraining
augment: 5
batchsize: 108  
size: 32  # crop size 
cam_size: 960 # Patch size for CAM visualization. If not given, full images are used.
backbone: spectformer-xs # can be 'unet' etc. Refer to
features: 32 # number of channels in the output of the model
train_samples_per_epoch: 50000  # resampling to balance the positive and negative classes
num_workers: 4  #  legacy
projhead: minimal_batchnorm
classhead: minimal
input_train:
# optional. Can specify path to training data here
split_train:
- - 0.0 # should be 0.0 and use all the training data available
  - 1.0
split_val:
- 0.0
- 1.0  # 0.1
```


#### Data preprocessing
```
CUDA_VISIBLE_DEVICES=1 python data_prep.py \
--input_frame /path/021221_C16_1_8bit_PFFC.tif \  # path to the movie frames to classify
--input_mask /path/021221_C16-1_8bit_PFFC-BrightAdj_annotated_classes.tif \ # path to the masks (i.e. labels) of the movie 
--data_save_dir /path/size_48_crops_1000_sample_500000_seed_25_area_0 \ # path to save the preprocessed data
--frames \ # number of input frames for each training sample.
--size 48 \  # crop size
--crops_per_image 1000 \ # number of crops taken from one frame
--data_seed 25 \ # random seed for generating crops
--pixel_area_threshold 0  \ # lower threshold of the pixel area for event labelling
--binary_problem True \ # Boolean indicating whether we want to formulate a binary or multi-class classification. Default: True  
```
For a comprehensive list of key word arguments and their definitions, please refer to the script "data_prep.py".

#### Event classification
```
CUDA_VISIBLE_DEVICES=1 python event_classification.py \
--input_frame /path/021221_C16_1_8bit_PFFC.tif \   
--input_mask /path/021221_C16-1_8bit_PFFC-BrightAdj_annotated_classes.tif \ 
--data_save_dir /path/preprocessed_data/size_48_crops_1000_sample_500000_seed_25_area_40 \
--model_save_dir /path \ # path to save trained event classification head 
--dataset_save_dir /path \ # directory to save train, validation and test data 
--TAP_model_load_path /path/06-04-15-26-30_synergy_backbone_unet \
--cls_head_arch resnet \
--model_id resnet_head_2024-12-02-1853_model_seed_45 \
--crops_per_image 1000 \
--balanced_sample_size 500000 \
--size 48 \
--batchsize 108 \
--training_epochs 30 \
--model_seed 45 \
--data_seed 25 \
--num_runs 1 \
--TAP_init loaded # 'loaded': load a pretrained weights; 'km_init': random weights using kaiming initialisation
```

#### Examine mistaken predictions
```
CUDA_VISIBLE_DEVICES=1 python examine_mistaken_predictions.py \
--masks_path /path/021221_C16-1_8bit_PFFC-BrightAdj_annotated_classes.tif \
--mistake_pred_dir /path \ # path to save the results
--TAP_model_load_path /path/06-04-15-26-30_synergy_backbone_unet \ # dense feature maps from TAP pretraining 
--test_data_load_path /path/test_data_crops_flat.pth \
--combined_model_load_dir /path \
--model_id resnet_head_2024-12-02-1853_model_seed_45 \ # model id for the classification head
--cls_head_arch resnet \ # architecture for the classfication head
--num_egs_to_show 0 \ # number of mistaken examples to export 
--is_true_positive \ # whether to export true positive examples
--is_true_negative # whether to export true negative examples
```
