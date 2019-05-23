#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

#### Stereo

### PSMNet
# create baseline PSMNet
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 1 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_PSMNet --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --logEvery 50 --testEvery 10 --saveEvery 200 --half
# try PSMNet with argumentation: scale from 2 to 0.5 (best)
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 1 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_PSMNet --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 2 0.7

### GwcNetGC
# create baseline GwcNetGC
#PYTHONPATH=./ python train/Stereo_train.py --model GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 1 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_GwcNetGC --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --logEvery 50 --testEvery 10 --saveEvery 200 --half
# submission: train with all trainset
#PYTHONPATH=./ python train/Stereo_train.py --model GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 1 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_sceneflow_GwcNetGC --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0001 --logEvery 50 --testEvery 0 --saveEvery 200 --half --subType trainSub



#### SRStereo

### PSMNet

## compare whether update SR
# without updating SR (better: no nan problem)
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half
# with updating SR
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights 0.5 0 0.5 --logEvery 50 --testEvery 10 --saveEvery 200 --half


## compare argumentation (without updating SR)
# noArg: SR was trained without argumentation
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline_noArg $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half
# allArg: SR was trained with argumentation and train Stereo with argumentation
# argumentation: scale from 1 to 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 1 0.5
# argumentation: scale from 2 to 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 2 0.5
# argumentation: scale from 2 to 1
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 2 1
# argumentation: scale from 3 to 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 3 0.5
# argumentation: scale from 1.5 to 0.75 (best)
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 1.5 0.75
# argumentation: scale from 1.75 to 0.63
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 1.75 0.63
# argumentation: scale from 1.3 to 0.77
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 1.3 0.77


## compare: learning rate (argumentation: scale from 1.5 to 0.75)
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.5 0.75
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint logs/experiments/compare_kitti_SRStereo_Stereo/Stereo_train/190504082252_model_SRStereo_EDSR_PSMNet_loadScale_1.0_trainCrop_64_512_batchSize_12_4_lossWeights_-1.0_0.0_1.0_kitti2015/checkpoint_epoch_0600_it_00014.tar --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 2400 --lr 0.001 300 0.0005 600 0.0002 900 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.5 0.75 --resume toNew
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint logs/experiments/compare_kitti_SRStereo_Stereo/Stereo_train/190504082252_model_SRStereo_EDSR_PSMNet_loadScale_1.0_trainCrop_64_512_batchSize_12_4_lossWeights_-1.0_0.0_1.0_kitti2015/checkpoint_epoch_0600_it_00014.tar --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 2400 --lr 0.001 300 0.0005 600 0.0002 900 0.0001 1200 0.00005 1500 0.00002 1800 0.00001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.5 0.75 --resume toNew
# result: no need training 2400 epochs and '0.001 300 0.0005 450 0.0002 600 0.0001' is good enough

## finnal KITTI train (SERVER 162)
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_PSMNet_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 2400 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 1.5 0.75

## finnal KITTI submission
# train
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline_trainSub $pretrained_kitti2015_PSMNet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 0 --saveEvery 200 --half --argument 1.5 0.75 --subType trainSub
# submission
#PYTHONPATH=./ python evaluation/Stereo_eval.py --model SRStereo EDSR PSMNet --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset_testing --dataset kitti2015 --chkpoint $pretrained_kitti2015_SRStereo_trainSub --loadScale 1 --batchSize 0 1 --evalFcn outlier --half --noComet --subType subTest --resume toOld


### GwcNetGC

## try different combinations
# compare: Stereo trained with carla
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_carla_GwcNetGCDown --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 1.5 0.75

## compare: argumentation
# noArg: SR was trained without argumentation
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half
# allArg: SR was trained with argumentation and train Stereo with argumentation
# argumentation: scale from 2 to 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 2 0.5
# argumentation: scale from 1 to 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 1 0.5
# argumentation: scale from 2 to 1
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 200 --half --argument 2 1
# argumentation: scale from 1.75 to 0.63 (lr_slowDecay, best)
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 600 0.0002 900 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.75 0.63
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint ${experiment_dir}/compare_kitti_SRStereo_Stereo/Stereo_train/190529091142_model_SRStereo_EDSR_GwcNetGC_loadScale_1.0_trainCrop_64_512_batchSize_12_4_lossWeights_-1.0_0.0_1.0_kitti2015 --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 2400 --lr 0.001 300 0.0005 600 0.0002 900 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.75 0.63 --resume toNew
# argumentation: scale from 1.75 to 0.63
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.75 0.63
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint ${experiment_dir}/compare_kitti_SRStereo_Stereo/Stereo_train/190530000836_model_SRStereo_EDSR_GwcNetGC_loadScale_1.0_trainCrop_64_512_batchSize_12_4_lossWeights_-1.0_0.0_1.0_kitti2015 --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 2400 --lr 0.001 300 0.0005 450 0.0002 600 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.75 0.63 --resume toNew

# compare: learining rate (argumentation: scale from 1.5 to 0.75)
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.5 0.75
# lr_slowDecay
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 600 0.0002 900 0.0001  --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.5 0.75
# lr_slowDecay, argumentation: scale from 2 to 0.5
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 300 0.0005 600 0.0002 900 0.0001  --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 2 0.5
# lr_fastDecay, argumentation: scale from 1.75 to 0.63
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 1200 --lr 0.001 200 0.0005 300 0.0002 400 0.0001  --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.75 0.63
# lr_sslowDecay, argumentation: scale from 1.75 to 0.63
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 2400 --lr 0.001 400 0.0005 800 0.0002 1200 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.75 0.63

# compare: batch size


## finnal KITTI train (SERVER 95)
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline $pretrained_kitti2015_GwcNetGC_trainSet --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 2400 --lr 0.001 300 0.0005 600 0.0002 900 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 10 --saveEvery 300 --half --argument 1.75 0.63

## finnal KITTI submission
# train (SERVER 199)
#PYTHONPATH=./ python train/Stereo_train.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset --dataset kitti2015 --chkpoint $pretrained_kitti2015_EDSRbaseline_trainSub $pretrained_kitti2015_GwcNetGC --loadScale 1 --batchSize 12 $nGPUs --trainCrop 64 512 --evalFcn outlier --epochs 2400 --lr 0.001 300 0.0005 600 0.0002 900 0.0001 --lossWeights -1 0 1 --logEvery 50 --testEvery 0 --saveEvery 300 --half --argument 1.75 0.63 --subType trainSub
# submission
PYTHONPATH=./ python evaluation/Stereo_eval.py --model SRStereo EDSR GwcNetGC --outputFolder experiments/compare_kitti_SRStereo_Stereo --dispScale 2 --dataPath $kitti2015_dataset_testing --dataset kitti2015 --chkpoint $pretrained_kitti2015_GwcNetGC_trainSub --loadScale 1 --batchSize 0 1 --evalFcn outlier --half --noComet --subType subTest --resume toOld
