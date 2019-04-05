#!/usr/bin/env bash

cd ../

# datasets
carla_kitti_dataset=../datasets/carla_kitti/carla_kitti_sr_lowquality/
carla_kitti_dataset_moduletest=../datasets/carla_kitti/carla_kitti_sr_lowquality_moduletest
carla_kitti_dataset_overfit=../datasets/carla_kitti/carla_kitti_sr_lowquality_overfit
sceneflow_dataset=../datasets/sceneflow/
kitti2015_dataset=../datasets/kitti/data_scene_flow/training/
kitti2015_sr_dataset=../datasets/kitti/data_scene_flow_sr/training/
kitti2015_dense_dataset=../datasets/kitti/data_scene_flow_dense/training/
kitti2012_dataset=../datasets/kitti/data_stereo_flow/training/

# dir setting
pretrained_dir=logs/pretrained
experiment_dir=logs/experiments
experiment_bak_dir=logs/experiments_bak

# pretrained models
pretrained_PSMNet_sceneflow=${pretrained_dir}/PSMNet_pretrained_sceneflow/PSMNet_pretrained_sceneflow.tar
pretrained_PSMNet_kitti2012=${pretrained_dir}/PSMNet_pretrained_model_KITTI2012/PSMNet_pretrained_model_KITTI2012.tar
pretrained_PSMNet_kitti2015=${pretrained_dir}/PSMNet_pretrained_model_KITTI2015/PSMNet_pretrained_model_KITTI2015.tar
pretrained_EDSR_DIV2K=${pretrained_dir}/EDSR_pretrained_DIV2K/EDSR_baseline_x2.pt

# python scripts
stereo_train=train/Stereo_train.py
stereo_eval=train/Stereo_eval.py
