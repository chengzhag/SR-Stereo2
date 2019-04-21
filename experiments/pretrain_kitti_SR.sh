#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# finetune SR
PYTHONPATH=./ python train/SR_train.py --model EDSR --outputFolder experiments/pretrain_kitti_SR --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_EDSR_DIV2K --batchSize 64 $(( 4 * $nGPUs)) --trainCrop 64 512 --evalFcn l1 --epochs 6000 --lr 0.0001 --logEvery 50 --testEvery 50 --saveEvery 1000 --half --argument
