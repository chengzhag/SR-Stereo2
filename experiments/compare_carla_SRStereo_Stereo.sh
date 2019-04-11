#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# SRStereo
# train EDSRbaselinePSMNetDown
PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_carla_SRStereo_Stereo --dispScale 2 --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_PSMNet_carla --loadScale 0.5 --batchSize 12 $nGPUs --trainCrop 128 1024 --evalFcn l1 --epochs 5 --lr 0.0001 --logEvery 50 --testEvery -5 --saveEvery 5 --half


