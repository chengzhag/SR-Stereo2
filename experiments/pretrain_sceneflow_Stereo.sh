#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

## Stereo
# train PSMNet (SERVER 95)
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 1 --saveEvery 1 --half --mask 1 1 1 0
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 2 --maxDisp 96 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 1 --saveEvery 1 --half --mask 1 1 1 0
# train GwcNetGC
#PYTHONPATH=./ python train/Stereo_train.py --model GwcNetGC --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 8 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 16 --lr 0.001 10 0.0005 12 0.00025 14 0.000125 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
#PYTHONPATH=./ python train/Stereo_train.py --model GwcNetGC --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 16 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 16 --lr 0.001 10 0.0005 12 0.00025 14 0.000125 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0 --half
# train GwcNetG
#PYTHONPATH=./ python train/Stereo_train.py --model GwcNetG --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 8 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 16 --lr 0.001 10 0.0005 12 0.00025 14 0.000125 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
# train GANet
#PYTHONPATH=./ python train/Stereo_train.py --model GANet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 4 $nGPUs --trainCrop 240 576 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 10 --saveEvery 1 --mask 1 1 1 0
# train MDSRfeaturePSMNetBody
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo MDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_MDSR --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --lossWeights 0 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0 --noComet
# train EDSRfeaturePSMNetBody
#(SERVER 95)
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --lossWeights 0 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
#(SERVER 162)
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint ${experiment_dir}/pretrain_sceneflow_Stereo/Stereo_train/190704112543_model_FeatureStereo_EDSRfeature_PSMNetBody_loadScale_1.0_trainCrop_256_512_batchSize_12_4_lossWeights_0.0_1.0_sceneflow/checkpoint_epoch_0005_it_02955.tar --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0001 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0 --resume toNew
PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0001 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0

