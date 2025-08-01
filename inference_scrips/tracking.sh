#!/bin/bash

# Model type is Weight Name
# Choices: POMATO_pairwise, POMATO_temp_6frames, POMATO_temp_12frames
MODEL_TYPE="POMATO_pairwise" 
WEIGHT="./pretrained_models/${MODEL_TYPE}.pth"

# input path is the tracking evaluation data path, whose subfolders are the specific evaluation dataset
# output path is the path to save the tracking results, which will be saved as xxx.npz files
# visualize output path is the path to save the visualization results
INPUT_PATH="./data/tracking_eval_data"
OUTPUT_PATH="./outputs/model_outputs/${MODEL_TYPE}"
VISUALIZ_OUTPUT_PATH="./outputs/visualize_tracking"

# Datasets to be evaluated
# For example, adt_seq_12_2 is adt dataset with sequence length 12 and sample step is 2 compared to the original dataset.
# DATASETS="adt_seq_12_2,adt_seq_24_2,pstudio_seq_12_2,pstudio_seq_24_2,po_seq_12_2,po_seq_24_2"   
DATASETS="adt_seq_12_2" 


# tracking the data
python ./inference_scrips/tracking.py \
    --model_type $MODEL_TYPE \
    --weights $WEIGHT \
    --device cuda:0 \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --dataset $DATASETS \
    --window 12 \
    --overlap 5


# evaluate the tracking results
python ./dust3r/tracking_eval.py \
    --gt_dir $INPUT_PATH \
    --pred_dir $OUTPUT_PATH \
    --dataset $DATASETS \

# visualize the tracking results
# dataset is the specific dataset to visualize, e.g., adt_seq_12_2
python ./dust3r/visualize_tracking.py \
    --model_dataset \
    --gt_dir $INPUT_PATH \
    --pred_dir $OUTPUT_PATH \
    --model_name $MODEL_TYPE \
    --output_path $VISUALIZ_OUTPUT_PATH \
    --dataset adt_seq_12_2 \
    --show_gt
