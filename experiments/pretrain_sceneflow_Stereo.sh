#!/usr/bin/env bash
. ./settings.sh

# GPU settings
export CUDA_VISIBLE_DEVICES=1,0,2,3
nGPUs=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

## Stereo

# train PSMNet
# PSMNet: create baseline
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 1 --saveEvery 1 --half --mask 1 1 1 0
# PSMNet_step_2: try creating cost volume with step 2
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 2 --maxDisp 96 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 1 --saveEvery 1 --half --mask 1 1 1 0
# PSMNet_load_PSMNetSR: load feature from PSMNetSR trained with DIV2K
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_PSMNetSR --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 1 --saveEvery 1 --half --mask 1 1 1 0
# PSMNet_loadPSMNetSRfhC: load feature from PSMNetSRfullHalfCat trained with DIV2K
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_PSMNetSRfullHalfCat --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 1 --saveEvery 1 --half --mask 1 1 1 0
# PSMNet_loadPSMNetSRfChR: load feature from PSMNetSRfullCatHalfRes trained with DIV2K
#PYTHONPATH=./ python train/Stereo_train.py --model PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_PSMNetSRfullCatHalfRes --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 1 --saveEvery 1 --half --mask 1 1 1 0


# train GwcNetGC
#PYTHONPATH=./ python train/Stereo_train.py --model GwcNetGC --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 8 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 16 --lr 0.001 10 0.0005 12 0.00025 14 0.000125 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
#PYTHONPATH=./ python train/Stereo_train.py --model GwcNetGC --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 16 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 16 --lr 0.001 10 0.0005 12 0.00025 14 0.000125 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0 --half

# train GwcNetG
#PYTHONPATH=./ python train/Stereo_train.py --model GwcNetG --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 8 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 16 --lr 0.001 10 0.0005 12 0.00025 14 0.000125 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0

# train GANet
#PYTHONPATH=./ python train/Stereo_train.py --model GANet --outputFolder experiments/pretrain_sceneflow_Stereo --dispScale 1 --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 4 $nGPUs --trainCrop 240 576 --evalFcn l1 --epochs 10 --lr 0.001 --logEvery 50 --testEvery 10 --saveEvery 1 --mask 1 1 1 0

## FeatureStereo

# train MDSRfeaturePSMNetBody
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo MDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_MDSR --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --lossWeights 0 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0 --noComet

# train EDSRfeaturePSMNetBody
# EDSRfeaturePSMNetBody_bodyUpdate: only update body
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 12 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --lossWeights 0 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
# EDSRfeaturePSMNetBody_allUpdate: update feature and body from epoch 5
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint ${experiment_dir}/pretrain_sceneflow_Stereo/Stereo_train/190704112543_model_FeatureStereo_EDSRfeature_PSMNetBody_loadScale_1.0_trainCrop_256_512_batchSize_12_4_lossWeights_0.0_1.0_sceneflow/checkpoint_epoch_0005_it_02955.tar --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0001 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0 --resume toNew
# EDSRfeaturePSMNetBody_allUpdate: update feature and body from begining
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0001 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
# EDSRfeaturePSMNetBody_noLoad_allUpdate: update feature and body from begining wihout loading SRfeature
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0001 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0

# train EDSRfeaturePSMNet
# EDSRfeaturePSMNet_stereoUpdate: only update stereo
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 8 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.001 --lossWeights 0 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
# EDSRfeaturePSMNet_allUpdate: update feature and stereo
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0001 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0

# EDSRfeaturePSMNetBody_allUpdate lr test
# EDSRfPSMNetB_lr_0.0005_allUpdate: NaN
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0005 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
# EDSRfPSMNetB_lr_0.0002_allUpdate: NaN
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNetBody --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0005 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0

# EDSRfeaturePSMNet_allUpdate lr test
# EDSRfPSMNet_lr_0.0005_allUpdate: NaN
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0005 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
# EDSRfPSMNet_lr_0.0002_allUpdate
#PYTHONPATH=./ python train/Stereo_train.py --model FeatureStereo EDSRfeature PSMNet --outputFolder experiments/pretrain_sceneflow_Stereo --dataPath $sceneflow_dataset --dataset sceneflow --chkpoint $pretrained_DIV2K_EDSR_baseline_x2 None --loadScale 1 --batchSize 4 $nGPUs --trainCrop 256 512 --evalFcn l1 --epochs 10 --lr 0.0002 --lossWeights 1 1 --logEvery 50 --testEvery 1 --saveEvery 1 --mask 1 1 1 0
