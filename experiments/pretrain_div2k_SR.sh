#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# train EDSR
PYTHONPATH=./ python train/SR_train.py --model EDSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.0001 250 0.00005 --logEvery 200 --testEvery 10 --saveEvery 50

