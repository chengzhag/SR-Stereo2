#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# SRStereoRefine
# train EDSRbaselinePSMNetRefine
PYTHONPATH=./ python train/Stereo_train.py --model SRStereoRefine EDSR PSMNet --outputFolder experiments/compare_carla_SRStereoRefine --dispScale 2 --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_carla_EDSRbaselineDisp $pretrained_carla_PSMNetDown --loadScale 1 0.5 --batchSize 12 $nGPUs --trainCrop 128 1024 --evalFcn l1 --epochs 5 --lr 0.0001 --lossWeights 0.5 0.375 0.125 --logEvery 50 --testEvery -5 --saveEvery 5 --half
