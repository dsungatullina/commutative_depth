#!/bin/sh
module load gpu/cuda-9.0
module load python/anaconda3
source activate python3.5-env 
python -m visdom.server -port 8091 &
python train.py \
--niter 15 \
--niter_decay 15 \
--display_port 8091 \
--batchSize 16 \
--gpu_ids 0 \
--name fake_kitti_base_crop_20 \
--model supervised \
--shuffle \
--img_source_file /gpfs/gpfs0/d.sungatullina/fake_kitti_20.txt \
--lab_source_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_depth_all.txt \
--img_target_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_rgb_test.txt \
--lab_target_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_depth_test.txt \
--resize \
--crop \
--flip \
--rotation \
