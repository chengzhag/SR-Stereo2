#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

## create baseline PSMNet
PYTHONPATH=./ python $stereo_train --model PSMNet --outputFolder experiments/SRStereo_Stereo1_compare_kitti --dispScale 1 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_PSMNet_sceneflow --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 300 --lr 0.001 200 0.0001 --logEvery 10 --testEvery 10 --saveEvery 30
