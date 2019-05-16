#!/bin/sh
module load gpu/cuda-9.0
module load python/anaconda3
source activate python3.5-env 
python -m visdom.server -port 8093 &
python train.py \
--niter 100 \
--niter_decay 100 \
--display_port 8093 \
--batchSize 32 \
--gpu_ids 0,1,2,3 \
--name 01_vkitti_base_fair_smooth_50 \
--model supervised \
--shuffle \
--img_source_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_rgb_all.txt \
--lab_source_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_depth_all.txt \
--img_target_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_rgb_test.txt \
--lab_target_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_depth_test.txt \
--resize \
--flip \
--rotation \
