#!/bin/sh
module load gpu/cuda-9.0
module load python/anaconda3
source activate python3.5-env
python -m visdom.server -port 8091 &
python train.py \
--niter 15 \
--niter_decay 15 \
--display_port 8091 \
--batchSize 4 \
--gpu_ids 0,1,2,3 \
--name clip50_l1-10.0_com-r-10.0_com-s-0.0_cycle-1.0 \
--model commutative \
--img_source_file /gpfs/gpfs0/d.sungatullina/Datasets/transfer-to-zhores/lists_512/synthia_rgb_train.txt \
--img_target_file /gpfs/gpfs0/d.sungatullina/Datasets/transfer-to-zhores/lists_512/cityscapes_rgb_train.txt \
--lab_source_file /gpfs/gpfs0/d.sungatullina/Datasets/transfer-to-zhores/lists_512/synthia_depth_train.txt \
--lab_target_file /gpfs/gpfs0/d.sungatullina/Datasets/transfer-to-zhores/lists_512/cityscapes_depth_train.txt \
--shuffle \
--crop \
--cropSize 256 \
--flip \
--rotation \
--init_models \
--init_Depth_netG_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/latest_net_img2task.pth \
--init_R2S_netG_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/5_net_G_B.pth \
--init_R2S_netD_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/5_net_D_B.pth \
--init_S2R_netG_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/5_net_G_A.pth \
--init_S2R_netD_filename /gpfs/gpfs0/d.sungatullina/commutative_depth/init_models/5_net_D_A.pth \
--com_loss usual \
--l1syndepth_loss usual \
--lambda_com_S 0.0 \
--lambda_com_R 10.0 \
--lambda_l1_DS 10.0 \
--lambda_S 10.0 \
--lambda_R 10.0 \
--lambda_cycle 1.0 \
--lr_task 0.0001 \
--lr_trans 0.00005 \
--beta1 0.5 \

