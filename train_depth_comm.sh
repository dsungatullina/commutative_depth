python -m visdom.server -port 8091 &
python train.py \
--niter 15 \
--niter_decay 15 \
--display_port 8091 \
--batchSize 1 \
--gpu_ids 0,1 \
--name tmp \
--model commutative \
--img_source_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/synthia_rgb_train.txt \
--img_target_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/cityscapes_rgb_train.txt \
--lab_source_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/synthia_depth_train.txt \
--lab_target_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/cityscapes_depth_train.txt \
--shuffle \
--crop \
--cropSize 256 \
--flip \
--rotation \
--init_models \
--init_Depth_netG_filename /media/hpc-4_Raid/dsungatullina/submission/commutative_depth/init_models/latest_net_img2task.pth \
--init_R2S_netG_filename /media/hpc-4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_G_B.pth \
--init_R2S_netD_filename /media/hpc-4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_D_B.pth \
--init_S2R_netG_filename /media/hpc-4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_G_A.pth \
--init_S2R_netD_filename /media/hpc-4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_D_A.pth \
--com_loss usual \
--l1syndepth_loss usual \
--lambda_com_S 5.0 \
--lambda_com_R 5.0 \
--lambda_l1_DS 10.0 \
--lambda_S 10.0 \
--lambda_R 10.0 \
--lambda_cycle 1.0 \
--lr_task 0.0001 \
--lr_trans 0.00005 \
--beta1 0.5 \

