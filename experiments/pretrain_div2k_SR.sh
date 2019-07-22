#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# (SERVER 162) train EDSR
#PYTHONPATH=./ python train/SR_train.py --model EDSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.0001 250 0.00005 --logEvery 200 --testEvery 10 --saveEvery 50

# (SERVER 11) train PSMNetSR
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.0001 250 0.00005 --logEvery 200 --testEvery 10 --saveEvery 50

# PSMNetSR lr test
# (SERVER 95)
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 200 --testEvery 10 --saveEvery 50
# (SERVER 162)
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.0005 250 0.00025 --logEvery 200 --testEvery 10 --saveEvery 50
