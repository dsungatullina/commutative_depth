import os
import glob
import cv2 

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tifffile import imsave
from scipy.ndimage.morphology import grey_closing

# train dir
filenames = glob.glob('/media/hpc4_Raid/data/SYNTHIA/RAND_CITYSCAPES/Depth/train/*.png')
filenames.sort()
print(len(filenames))
for i in range(len(filenames)):
    name = filenames[i].split('/')[-1]
    depth = cv2.imread(filenames[i], cv2.IMREAD_UNCHANGED)
    depth = depth[:, :, 0] / 100.0
    depth = cv2.resize(depth, (512, 304), interpolation=cv2.INTER_NEAREST) 
    imsave('/media/hpc4_Raid/dsungatullina/transfer-to-zhores/SYNTHIA/RAND_CITYSCAPES/Depth_512x304/train/' + name.split('.')[-2] + '.tif', depth)

# test dir
filenames = glob.glob('/media/hpc4_Raid/data/SYNTHIA/RAND_CITYSCAPES/Depth/test/*.png')
filenames.sort()
print(len(filenames))
for i in range(len(filenames)):
    name = filenames[i].split('/')[-1]
    depth = cv2.imread(filenames[i], cv2.IMREAD_UNCHANGED)
    depth = depth[:, :, 0] / 100.0
    depth = cv2.resize(depth, (512, 304), interpolation=cv2.INTER_NEAREST) 
    imsave('/media/hpc4_Raid/dsungatullina/transfer-to-zhores/SYNTHIA/RAND_CITYSCAPES/Depth_512x304/test/' + name.split('.')[-2] + '.tif', depth)

# val dir
filenames = glob.glob('/media/hpc4_Raid/data/SYNTHIA/RAND_CITYSCAPES/Depth/val/*.png')
filenames.sort()
print(len(filenames))
for i in range(len(filenames)):
    name = filenames[i].split('/')[-1]
    depth = cv2.imread(filenames[i], cv2.IMREAD_UNCHANGED)
    depth = depth[:, :, 0] / 100.0
    depth = cv2.resize(depth, (512, 304), interpolation=cv2.INTER_NEAREST) 
    imsave('/media/hpc4_Raid/dsungatullina/transfer-to-zhores/SYNTHIA/RAND_CITYSCAPES/Depth_512x304/val/' + name.split('.')[-2] + '.tif', depth)


