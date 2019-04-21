#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

## Stereo
# create baseline PSMNet
PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 1 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_PSMNet_sceneflow --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --logEvery 50 --testEvery 10 --saveEvery 200 --half

## SRStereo
# TODO
