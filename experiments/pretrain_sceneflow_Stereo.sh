#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

## Stereo
# train PSMNet
PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 1 --saveEvery 1 --half --mask 1 1 1 0
# train GwcNetGC
#PYTHONPATH=./ python train/Stereo_train.py --model GwcNetGC --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 8 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 16 --lr 0.001 10 0.0005 12 0.00025 14 0.000125 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
PYTHONPATH=./ python train/Stereo_train.py --model GwcNetGC --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 16 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 16 --lr 0.001 10 0.0005 12 0.00025 14 0.000125 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0 --half
