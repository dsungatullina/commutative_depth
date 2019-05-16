#!/bin/sh
module load gpu/cuda-9.0
module load python/anaconda3
source activate python3.5-env 
python -m visdom.server -port 8092 &
python train.py \
--niter 10 \
--niter_decay 10 \
--display_port 8092 \
--batchSize 16 \
--gpu_ids 0,1 \
--name vkitti_base_fair \
--model supervised \
--shuffle \
--img_source_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_rgb_train.txt \
--lab_source_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_depth_train.txt \
--img_target_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_rgb_test.txt \
--lab_target_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_depth_test.txt \
--resize \
--flip \
--rotation \
