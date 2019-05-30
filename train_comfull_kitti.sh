python -m visdom.server -port 8092 &
python train_comseg_kitti.py \
--seg_model_name drn26 \
--lr_seg 0.0001 \
--niter 1 \
--niter_decay 9 \
--display_port 8092 \
--batchSize 1 \
--gpu_ids 0 \
--name test_comfull_model \
--model commutative_full \
--shuffle \
--img_target_file  /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/kitti/lists_hpc4/eigen_rgb_train.txt \
--img_source_file  /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/vkitti/lists_hpc4/vkitti_rgb_all.txt \
--lab_source_file  /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/vkitti/lists_hpc4/vkitti_labels_all.txt \
--lab_target_file '' \
--resize --loadSize 192,640 \
--flip --rotation \
--init_models \
--init_R2S_netG_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_G_B.pth \
--init_R2S_netD_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_D_B.pth \
--init_S2R_netG_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_G_A.pth \
--init_S2R_netD_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/5_net_D_A.pth \
--init_seg_netG_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/vkitti_drn26-iter100000.pth \
--init_depth_netG_filename /media/hpc4_Raid/dsungatullina/submission/commutative_depth/init_models/30_net_img2depth.pth \

