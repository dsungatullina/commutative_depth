import os
import glob
import cv2 

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tifffile import imsave
from scipy.ndimage.morphology import grey_closing

# train dir
save_path = '/media/hpc4_Raid/dsungatullina/transfer-to-zhores/Cityscapes/depth_512x256/train/'
save_path_2 = '/media/hpc4_Raid/dsungatullina/depth_2048x1024/train/'
f = open('/media/hpc4_Raid/dsungatullina/train_disparity.txt')
for line in f:
    name = line[:-1].split('/')[-1]
    city_dir = line[:-1].split('/')[-2]
    if not os.path.exists(save_path + city_dir):
        os.makedirs(save_path + city_dir)
    if not os.path.exists(save_path_2 + city_dir):
        os.makedirs(save_path_2 + city_dir)
    disparity = cv2.imread(line[:-1], cv2.IMREAD_UNCHANGED).astype(np.float32)
    disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256.0
    depth = np.zeros(disparity.shape)
    depth[disparity>0] = (0.209313 * 2262.52) / disparity[disparity>0]
    depth = grey_closing(depth, size=(7,7))
    depth_resized = cv2.resize(depth, (512, 256), interpolation=cv2.INTER_NEAREST) 
    imsave(save_path_2 + city_dir + '/' + name.split('.')[-2] + '.tif', depth) 
    imsave(save_path + city_dir + '/' + name.split('.')[-2] + '.tif', depth_resized)
f.close()

# test dir
save_path = '/media/hpc4_Raid/dsungatullina/transfer-to-zhores/Cityscapes/depth_512x256/test/'
save_path_2 = '/media/hpc4_Raid/dsungatullina/depth_2048x1024/test/'
f = open('/media/hpc4_Raid/dsungatullina/test_disparity.txt')
for line in f:
    name = line[:-1].split('/')[-1]
    city_dir = line[:-1].split('/')[-2]
    if not os.path.exists(save_path + city_dir):
        os.makedirs(save_path + city_dir)
    if not os.path.exists(save_path_2 + city_dir):
        os.makedirs(save_path_2 + city_dir)
    disparity = cv2.imread(line[:-1], cv2.IMREAD_UNCHANGED).astype(np.float32)
    disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256.0
    depth = np.zeros(disparity.shape)
    depth[disparity>0] = (0.209313 * 2262.52) / disparity[disparity>0]
    depth = grey_closing(depth, size=(7,7))
    depth_resized = cv2.resize(depth, (512, 256), interpolation=cv2.INTER_NEAREST) 
    imsave(save_path_2 + city_dir + '/' + name.split('.')[-2] + '.tif', depth) 
    imsave(save_path + city_dir + '/' + name.split('.')[-2] + '.tif', depth_resized)
f.close()

# val dir
save_path = '/media/hpc4_Raid/dsungatullina/transfer-to-zhores/Cityscapes/depth_512x256/val/'
save_path_2 = '/media/hpc4_Raid/dsungatullina/depth_2048x1024/val/'
f = open('/media/hpc4_Raid/dsungatullina/val_disparity.txt')
for line in f:
    name = line[:-1].split('/')[-1]
    city_dir = line[:-1].split('/')[-2]
    if not os.path.exists(save_path + city_dir):
        os.makedirs(save_path + city_dir)
    if not os.path.exists(save_path_2 + city_dir):
        os.makedirs(save_path_2 + city_dir)
    disparity = cv2.imread(line[:-1], cv2.IMREAD_UNCHANGED).astype(np.float32)
    disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256.0
    depth = np.zeros(disparity.shape)
    depth[disparity>0] = (0.209313 * 2262.52) / disparity[disparity>0]
    depth = grey_closing(depth, size=(7,7))
    depth_resized = cv2.resize(depth, (512, 256), interpolation=cv2.INTER_NEAREST) 
    imsave(save_path_2 + city_dir + '/' + name.split('.')[-2] + '.tif', depth) 
    imsave(save_path + city_dir + '/' + name.split('.')[-2] + '.tif', depth_resized)
f.close()
