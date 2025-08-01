#! /bin/bash
INPUT_DIR=assets/car
OUTPUT_DIR=./recon_results/car
MOVING_DISTANCE_THRES_GLOBAL=0.2            # Decrease this value to reduce the regions involved in the flow loss during the global alignment.
MOVING_DISTANCE_THRES_ADJACENT=0.4          # Decrease this value to seperate more regions as moving regions.
FLOW_LOSS_WEIGHT=0.01                       # set 0 to disable flow loss
WEIGHT='pretrained_models/POMATO_pairwise.pth'

python demo.py --input ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --weight ${WEIGHT} --moving_distance_thres_global ${MOVING_DISTANCE_THRES_GLOBAL} \
    --moving_distance_thres_adjacent ${MOVING_DISTANCE_THRES_ADJACENT} --flow_loss_weight ${FLOW_LOSS_WEIGHT}
python viser/visualizer_monst3r.py --data ${OUTPUT_DIR}/NULL --conf_thre 0.1