#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# StereoDown
# train PSMNetDown
#PYTHONPATH=./ python train/Stereo_train.py --model StereoDown PSMNet --outputFolder experiments/compare_carla_Stereo_StereoDown --dispScale 2 --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_carla_PSMNetDown --loadScale 1 0.5 --batchSize 12 $nGPUs --trainCrop 128 1024 --evalFcn l1 --epochs 5 --lr 0.0001 --lossWeights 0.75 0.25 --logEvery 50 --testEvery -5 --saveEvery 5 --half
PYTHONPATH=./ python train/Stereo_train.py --model StereoDown PSMNet --outputFolder experiments/compare_carla_Stereo_StereoDown --dispScale 2 --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_carla_PSMNetDown --loadScale 1 0.5 --batchSize 4 $nGPUs --trainCrop 256 1536 --evalFcn l1 --epochs 5 --lr 0.0001 --lossWeights 0.75 0.25 --logEvery 50 --testEvery -5 --saveEvery 5 --half

# Stereo
# train PSMNet
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/compare_carla_Stereo_StereoDown --dispScale 1 --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_carla_PSMNet --loadScale 0.5 --batchSize 12 $nGPUs --trainCrop 128 1024 --evalFcn l1 --epochs 5 --lr 0.0001 --logEvery 50 --testEvery -5 --saveEvery 5 --half

