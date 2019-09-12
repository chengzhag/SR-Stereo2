#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=1,0,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

# train EDSR
#PYTHONPATH=./ python train/SR_train.py --model EDSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 6000 --lr 0.0001 4000 0.00005 --logEvery 500 --testEvery 200 --saveEvery 1000

# train PSMNetSR
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 512 --evalFcn psnr --epochs 6000 --lr 0.001 4000 0.0005 --logEvery 500 --testEvery 200 --saveEvery 1000

# PSMNetSR lr test
# lr starts from 0.0001
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 6000 --lr 0.0001 4000 0.00005 --logEvery 500 --testEvery 200 --saveEvery 1000
# lr starts from 0.0005
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 6000 --lr 0.0005 4000 0.00025 --logEvery 500 --testEvery 200 --saveEvery 1000
# lr starts from 0.001 (better)
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 6000 --lr 0.001 4000 0.0005 --logEvery 500 --testEvery 200 --saveEvery 1000
# lr starts from 0.005
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 6000 --lr 0.005 4000 0.0025 --logEvery 500 --testEvery 200 --saveEvery 1000

# train PSMNetSRfullHalfCat
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullHalfCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 6000 --lr 0.001 4000 0.0005 --logEvery 500 --testEvery 200 --saveEvery 1000
# PSMNetSRfhC_crop256: cropsize 256
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullHalfCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 256 --evalFcn psnr --epochs 6000 --lr 0.001 4000 0.0005 --logEvery 500 --testEvery 200 --saveEvery 1000
# PSMNetSRfhC_crop512: cropsize 512
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullHalfCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 512 --evalFcn psnr --epochs 6000 --lr 0.001 4000 0.0005 --logEvery 500 --testEvery 200 --saveEvery 1000

# train PSMNetSRfullCatHalfRes
# PSMNetSRfChR_noDilated: no dilated convolution
# PSMNetSRfChR_avgBN: momentum set to None
# PSMNetSRfChR_smootherBN: momentum set to 0.01
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 512 --evalFcn psnr --epochs 6000 --lr 0.001 4000 0.0005 --logEvery 500 --testEvery 200 --saveEvery 1000
# PSMNetSRfChR_crop256: cropsize 256
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 256 --evalFcn psnr --epochs 6000 --lr 0.001 4000 0.0005 --logEvery 500 --testEvery 200 --saveEvery 1000
# PSMNetSRfChR_crop512: cropsize 512
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 512 --evalFcn psnr --epochs 6000 --lr 0.001 4000 0.0005 --logEvery 500 --testEvery 200 --saveEvery 1000

# train PSMNetSRfullCat
#PYTHONPATH=./ python train/SR_train.py --model PSMNetSRfullCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 --trainCrop 96 --evalFcn psnr --epochs 6000 --lr 0.001 4000 0.0005 --logEvery 500 --testEvery 200 --saveEvery 1000

# submission
##190731111452:
#PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 1 --chkpoint ${experiment_dir}/pretrain_div2k_SR/SR_train/190731111452_model_PSMNetSRfullCatHalfRes_loadScale_1_trainCrop_96_batchSize_16_lossWeights_1_DIV2K --evalFcn psnr --resume toOld --subType subEval --noComet
##190730150154:
#PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 1 --chkpoint ${experiment_dir}/pretrain_div2k_SR/SR_train/190730150154_model_PSMNetSRfullCatHalfRes_loadScale_1_trainCrop_96_batchSize_16_lossWeights_1_DIV2K --evalFcn psnr --resume toOld --subType subEval --noComet
##190729132606:
#PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 1 --chkpoint ${experiment_dir}/pretrain_div2k_SR/SR_train/190729132606_model_PSMNetSRfullCatHalfRes_loadScale_1_trainCrop_512_batchSize_16_lossWeights_1_DIV2K --evalFcn psnr --resume toOld --subType subEval --noComet
##190726185245:
#PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 1 --chkpoint ${experiment_dir}/pretrain_div2k_SR/SR_train/190726185245_model_PSMNetSRfullCatHalfRes_loadScale_1_trainCrop_256_batchSize_16_lossWeights_1_DIV2K --evalFcn psnr --resume toOld --subType subEval --noComet
##190726102730:
#PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSRfullHalfCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 1 --chkpoint ${experiment_dir}/pretrain_div2k_SR/SR_train/190726102730_model_PSMNetSRfullHalfCat_loadScale_1_trainCrop_96_batchSize_16_lossWeights_1_DIV2K --evalFcn psnr --resume toOld --subType subEval --noComet
#190724163918:
#PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 1 --chkpoint ${experiment_dir}/pretrain_div2k_SR/SR_train/190724163918_model_PSMNetSRfullCatHalfRes_loadScale_1_trainCrop_96_batchSize_16_lossWeights_1_DIV2K --evalFcn psnr --resume toOld --subType subEval --noComet
#190801193707:
#PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSRfullCatHalfRes --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 1 --chkpoint ${experiment_dir}/pretrain_div2k_SR/SR_train/190801193707_model_PSMNetSRfullCatHalfRes_loadScale_1_trainCrop_96_batchSize_16_lossWeights_1_DIV2K --evalFcn psnr --resume toOld --subType subEval --noComet
#190722193216:
#PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSR --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 1 --chkpoint ${experiment_dir}/pretrain_div2k_SR/SR_train/190722193216_model_PSMNetSR_loadScale_1_trainCrop_96_batchSize_16_lossWeights_1_DIV2K --evalFcn psnr --resume toOld --subType subEval --noComet
#190726102730:
#PYTHONPATH=./ python evaluation/SR_eval.py --model PSMNetSRfullHalfCat --outputFolder experiments/pretrain_div2k_SR --dataPath $sr_dataset --dataset DIV2K --batchSize 16 1 --chkpoint ${experiment_dir}/pretrain_div2k_SR/SR_train/190726102730_model_PSMNetSRfullHalfCat_loadScale_1_trainCrop_96_batchSize_16_lossWeights_1_DIV2K --evalFcn psnr --resume toOld --subType subEval --noComet


