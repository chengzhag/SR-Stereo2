#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

## FeatureStereo

# train EDSRfeaturePSMNetBody
# only update body (SERVER 95)
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNetBody_bodyUpdate --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --lossWeights 0 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0
# update feature and body (SERVER 162)
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNetBody_allUpdate --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.0001 200 0.00001 --lossWeights 1 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0

# train EDSRfeaturePSMNet
# only update body (SERVER 162)
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNet_stereoUpdate --loadScale 1 --batchSize 8 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --lossWeights 0 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0
# update feature and body (SERVER 95)
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNet_allUpdate --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.0001 200 0.00001 --lossWeights 1 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0


