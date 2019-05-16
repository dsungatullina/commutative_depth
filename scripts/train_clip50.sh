#!/bin/sh
module load gpu/cuda-9.0
module load python/anaconda3
source activate python3.5-env 
python -m visdom.server -port 8092 &
python train.py \
--niter 15 \
--niter_decay 15 \
--display_port 8092 \
--batchSize 20 \
--gpu_ids 0,1,2,3 \
--name clip50 \
--model supervised \
--shuffle \
--img_source_file /gpfs/gpfs0/d.sungatullina/Datasets/transfer-to-zhores/lists_512/synthia_rgb_train.txt \
--lab_source_file /gpfs/gpfs0/d.sungatullina/Datasets/transfer-to-zhores/lists_512/synthia_depth_train.txt \
--img_target_file /gpfs/gpfs0/d.sungatullina/Datasets/transfer-to-zhores/lists_512/cityscapes_rgb_train.txt \
--lab_target_file /gpfs/gpfs0/d.sungatullina/Datasets/transfer-to-zhores/lists_512/cityscapes_depth_train.txt \
--crop \
--cropSize 256 \
--flip \
--rotation \
