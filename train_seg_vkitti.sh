python -m visdom.server -port 8099 &
python train_seg_vkitti.py \
--niter 15 \
--niter_decay 15 \
--display_port 8099 \
--batchSize 1 \
--gpu_ids 0 \
--name test_run \
--model supervised_seg \
--shuffle \
--img_target_file  /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/kitti/lists_hpc4/eigen_rgb_train.txt \
--img_source_file  /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/vkitti/lists_hpc4/vkitti_rgb_all.txt \
--lab_source_file  /media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/vkitti/lists_hpc4/vkitti_labels_all.txt \
--lab_target_file '' \
--resize --loadSize 192,640 \
--flip --rotation \
