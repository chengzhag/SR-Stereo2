#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

## StereoDown
# train PSMNetDown
PYTHONPATH=./ python train/Stereo_train.py --model PSMNetDown --outputFolder experiments/pretrain_Stereo_StereoDown --dispScale 2 --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_PSMNet_sceneflow --loadScale 1 0.5 --batchSize 12 $nGPUs --trainCrop 128 1024 --evalFcn l1 --epochs 10 --lr 0.001 4 0.0005 6 0.00025 8 0.000125 --lossWeights 0.75 0.25 --logEvery 50 --testEvery 2 --saveEvery 2 --half

## Stereo
# train PSMNet
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/pretrain_Stereo_StereoDown --dispScale 2 --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_PSMNet_sceneflow --loadScale 0.5 --batchSize 12 $nGPUs --trainCrop 128 1024 --evalFcn l1 --epochs 10 --lr 0.001 4 0.0005 6 0.00025 8 0.000125 --logEvery 50 --testEvery 2 --saveEvery 2 --half

