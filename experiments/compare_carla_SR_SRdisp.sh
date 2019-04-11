#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# finetune SRdisp
PYTHONPATH=./ python train/SR_train.py --model EDSRdisp --outputFolder experiments/compare_carla_SR_SRdisp --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_EDSR_DIV2K --batchSize 16 $(( 2 * $nGPUs)) --trainCrop 64 2040 --evalFcn l1 --epochs 40 --lr 0.0005 15 0.0002 20 0.0001 25 0.00005 30 0.00002 35 0.00001 --logEvery 50 --testEvery 5 --saveEvery 40 --validSetSample 1 --half

# finetune SR
PYTHONPATH=./ python train/SR_train.py --model EDSR --outputFolder experiments/compare_carla_SR_SRdisp --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_EDSR_DIV2K --batchSize 16 $(( 2 * $nGPUs)) --trainCrop 64 2040 --evalFcn l1 --epochs 40 --lr 0.0001 25 0.00005 30 0.00002 35 0.00001 --logEvery 50 --testEvery 5 --saveEvery 40 --validSetSample 1 --half

