python -m visdom.server -port 8094 &
python train_comfull_kitti.py \
--name comfull_model_8094 \
--model commutative_full \
--gpu_ids 0 \
--batchSize 1 \
--display_port 8094 \
--seg_model_name drn26 \
--niter 1 \
--niter_decay 9 \
--img_target_file /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/kitti/lists_hpc4/eigen_rgb_train.txt \
--lab_target_file '' \
--img_source_file /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/vkitti/lists_hpc4/vkitti_rgb_all.txt \
--lab_source_file /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/vkitti/lists_hpc4/vkitti_labels_all.txt \
--dep_source_file /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/vkitti/lists_hpc4/vkitti_depth_all.txt \
--shuffle \
--resize --loadSize 192,640 \
--flip --rotation \
--init_models \
--init_R2S_netG_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_G_B.pth \
--init_R2S_netD_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_D_B.pth \
--init_S2R_netG_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_G_A.pth \
--init_S2R_netD_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_D_A.pth \
--init_seg_netG_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/vkitti_drn26-iter100000.pth \
--init_depth_netG_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/30_net_img2depth.pth \
--lr_seg 0.0001 \
--lr_dep 0.0001 \
--lr_trans 0.00005 \
--beta1 0.5 \
--lambda_S 10.0 \
--lambda_R 10.0 \
--lambda_cycle 1.0 \
--lambda_dep_S 50.0 \
--lambda_dep_com_S 10.0 \
--lambda_dep_com_R 0.0 \
--lambda_seg_S 50.0 \
--lambda_seg_com_S 10.0 \
--lambda_seg_com_R 0.0 \


