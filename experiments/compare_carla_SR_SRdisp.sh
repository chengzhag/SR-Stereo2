#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# finetune SRdisp
#PYTHONPATH=./ python train/SR_train.py --model EDSRdisp --outputFolder experiments/compare_carla_SR_SRdisp --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_EDSR_DIV2K --batchSize 4 $(( 2 * $nGPUs)) --trainCrop 96 1360 --evalFcn l1 --epochs 20 --lr 0.0001 10 0.00005 15 0.00002 --logEvery 50 --testEvery 2 --saveEvery 20 --validSetSample 1 --half

# finetune SR
#PYTHONPATH=./ python train/SR_train.py --model EDSR --outputFolder experiments/compare_carla_SR_SRdisp --dataPath $carla_dataset --dataset carla --chkpoint $pretrained_EDSR_DIV2K --batchSize 4 $(( 2 * $nGPUs)) --trainCrop 96 1360 --evalFcn l1 --epochs 20 --lr 0.0001 10 0.00005 15 0.00002 --logEvery 50 --testEvery 2 --saveEvery 20 --validSetSample 1 --half

