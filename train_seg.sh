python train_seg.py \
--display_id -1 \
--no_html \
--niter 15 \
--niter_decay 15 \
--display_port 8093 \
--batchSize 16 \
--gpu_ids 0,1 \
--name seg__fcn8_cityscapes_supervised_512x304_0.1 \
--model supervised_seg \
--shuffle \
--lr_task 0.1 \
--img_source_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/cityscapes_rgb_train.txt \
--lab_source_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/cityscapes_labels_train.txt \
--img_target_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/cityscapes_rgb_val.txt \
--lab_target_file /media/hpc-4_Raid/dsungatullina/transfer-to-zhores/_lists_512/cityscapes_labels_val.txt  \
--crop \
--cropSize 256 \
--flip \
--rotation \
