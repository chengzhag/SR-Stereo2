#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=1,0,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# train PSMNetSR
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_sceneflow_SR --dataPath $sceneflow_dataset --dataset sceneflow --batchSize 16 16 --trainCrop 512 512 --evalFcn psnr --epochs 135 --lr 0.001 90 0.0005 --logEvery 500 --testEvery 5 --saveEvery 30  --mask 1 0 0 0
