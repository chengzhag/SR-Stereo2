#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# train EDSR
#PYTHONPATH=./ python train/SR_train.py --model EDSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.0001 250 0.00005 --logEvery 500 --testEvery 10 --saveEvery 50

# train PSMNetSR
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 500 --testEvery 10 --saveEvery 50

# PSMNetSR lr test
# lr starts from 0.0001
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.0001 250 0.00005 --logEvery 500 --testEvery 10 --saveEvery 50
# lr starts from 0.0005
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.0005 250 0.00025 --logEvery 500 --testEvery 10 --saveEvery 50
# lr starts from 0.001 (better)
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 500 --testEvery 10 --saveEvery 50
# lr starts from 0.005
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.005 250 0.0025 --logEvery 500 --testEvery 10 --saveEvery 50

# train PSMNetSRfullHalfCat
# (SERVER 11)
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullHalfCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 500 --testEvery 10 --saveEvery 50
# PSMNetSRfhC_crop256: cropsize 256
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullHalfCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 256 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 500 --testEvery 10 --saveEvery 50
# PSMNetSRfhC_crop512: cropsize 512
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullHalfCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 512 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 500 --testEvery 10 --saveEvery 50

# train PSMNetSRfullCatHalfRes
# (SERVER 11 no dilated convolution)
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 500 --testEvery 10 --saveEvery 50
# PSMNetSRfChR_crop_256: cropsize 256
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 256 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 500 --testEvery 10 --saveEvery 50
# (SERVER 95) PSMNetSRfChR_crop_512: cropsize 512
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 512 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 500 --testEvery 10 --saveEvery 50

# train PSMNetSRfullCat
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 375 --lr 0.001 250 0.0005 --logEvery 500 --testEvery 10 --saveEvery 50

# submission
PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSRfullHalfCat --outputFolder submission/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 1 1 --chkpoint $pretrained_DIV2K_PSMNetSRfullHalfCat --evalFcn psnr --resume toNew --subType subEval --noComet

