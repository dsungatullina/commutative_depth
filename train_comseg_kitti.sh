#!/bin/sh
module load gpu/cuda-9.0
module load python/anaconda3
source activate python3.5-env
python -m visdom.server -port 8099 &
python train_comseg_kitti.py \
--seg_model_name drn26 \
--lr_seg 0.0001 \
--niter 1 \
--niter_decay 9 \
--display_port 8099 \
--batchSize 2 \
--gpu_ids 0 \
--name comseg_model \
--model commutative_seg \
--shuffle \
--img_target_file /gpfs/gpfs0/d.sungatullina/Datasets/kitti/lists_l/eigen_rgb_train.txt \
--img_source_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_rgb_all.txt \
--lab_target_file '' \
--lab_source_file /gpfs/gpfs0/d.sungatullina/Datasets/vkitti/lists/vkitti_labels_all.txt \
--resize --loadSize 192,640 \
--flip --rotation \
--init_models \
--init_R2S_netG_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/30_net_G_B.pth \
--init_R2S_netD_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/30_net_D_B.pth \
--init_S2R_netG_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/30_net_G_A.pth \
--init_S2R_netD_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/30_net_D_A.pth \
--init_seg_netG_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/vkitti_drn26-iter100000.pth \
--lambda_seg_S 10.0 \
--lambda_seg_com_S 1.0 \
--lambda_seg_com_R 1.0 \
