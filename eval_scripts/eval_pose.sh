#!/usr/bin/ bash

# Bonn dataset
torchrun --nproc_per_node=1 --master_port=29605 launch.py --mode=eval_pose \
      --pretrained="./pretrained_models/POMATO_pairwise.pth" \
      --eval_dataset=bonn --output_dir="results/bonn_pose" --pose_end_frame=40 \
      --pose_eval_stride=1 --moving_distance_thres_global=0.5 --moving_distance_thres_adjacent=0.4 --flow_loss_weight=0.01 \
      --temporal_smoothing_weight=0.5 --flow_loss_thre=40

# TUM dataset
torchrun --nproc_per_node=1 --master_port=29604 launch.py --mode=eval_pose \
      --pretrained="./pretrained_models/POMATO_pairwise.pth" \
      --eval_dataset=tum --output_dir="results/tum_pose" \
      --pose_end_frame=40 --pose_eval_stride=3 --moving_distance_thres_global=0.5 --moving_distance_thres_adjacent=0.4 --flow_loss_weight=0.01 \
      --temporal_smoothing_weight=0.5 --flow_loss_thre=25