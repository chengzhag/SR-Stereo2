#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

## FeatureStereo

# train EDSRfeaturePSMNetBody
# EDSRfeaturePSMNetBody_bodyUpdate: only update body
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNetBody_bodyUpdate --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --lossWeights 0 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0
# EDSRfeaturePSMNetBody_allUpdate: update feature and body
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNetBody_allUpdate --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.0001 200 0.00001 --lossWeights 1 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0

# train EDSRfeaturePSMNet
# EDSRfeaturePSMNet_stereoUpdate: only update stereo
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNet_stereoUpdate --loadScale 1 --batchSize 8 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --lossWeights 0 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0
# EDSRfeaturePSMNet_pretrainedSR_stereoUpdate: load EDSR pretrained with kitti and only update body
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_sceneflow_EDSRfeaturePSMNet_stereoUpdate --loadScale 1 --batchSize 8 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --lossWeights 0 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0
# EDSRfeaturePSMNet_allUpdate: update feature and body
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNet_allUpdate --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.0001 200 0.00001 --lossWeights 1 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0

# EDSRfeaturePSMNet_allUpdate lr test
# EDSRfPSMNet_lr_0.0005_allUpdate (better)
PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNet_allUpdate --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.0005 200 0.00005 --lossWeights 1 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0
# EDSRfPSMNet_lr_0.0002_allUpdate
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNet_allUpdate --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.0002 200 0.00002 --lossWeights 1 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0

# EDSRfeaturePSMNetBody_allUpdate lr test
# EDSRfPSMNetB_lr_0.0005_allUpdate
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNetBody_allUpdate --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.0005 200 0.00005 --lossWeights 1 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0
# EDSRfPSMNetB_lr_0.0002_allUpdate
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/compare_kitti_FeatureStereo --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_EDSRfeaturePSMNetBody_allUpdate --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.0002 200 0.00002 --lossWeights 1 1 --logEvery 50 --testEvery 50 --saveEvery 200 --mask 1 1 1 0
