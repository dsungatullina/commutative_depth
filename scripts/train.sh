#!/bin/sh
module load gpu/cuda-9.0
module load python/anaconda3
source activate python3.5-env 
python train.py \
--niter 15 \
--niter_decay 15 \
--display_port 8092 \
--batchSize 20 \
--gpu_ids 0,1 \
--name test_run \
--model supervised \
--shuffle \
--img_source_file /media/hpc-4_Raid/dsungatullina/submission/pytorch-CycleGAN-and-pix2pix/fake_cityscapes512x304_epoch5/_lists/fakecityscapes_rgb_train.txt \
--img_target_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/cityscapes_rgb_val.txt \
--lab_source_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/synthia_depth_train.txt \
--lab_target_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/cityscapes_depth_val.txt  \
--crop \
--cropSize 256 \
--flip \
--rotation \
