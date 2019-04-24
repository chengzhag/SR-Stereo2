#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

## Stereo
# create baseline PSMNet
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 1 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_PSMNet --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --logEvery 50 --testEvery 10 --saveEvery 200 --half

## SRStereo
# without updating SR
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half
# with updating SR
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.0001 --lossWeights 0.5 0 0.5 --logEvery 50 --testEvery 10 --saveEvery 200 --half
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights 0.5 0 0.5 --logEvery 50 --testEvery 10 --saveEvery 200 --half

# if using SR trained without argumentation
# without updating SR
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline_noArg $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half

# if training SRStereo with argumentation
# without updating SR
PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument

