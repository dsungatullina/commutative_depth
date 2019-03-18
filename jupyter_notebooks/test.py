import os
import glob
import cv2 

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tifffile import imsave
from scipy.ndimage.morphology import grey_closing

img_path = '/media/hpc4_Raid/data/SYNTHIA/RAND_CITYSCAPES/Depth/train/0000000.png'
depth = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
depth = depth[:, :, 0] / 100.0

imsave(img_path.split('/')[-1].split('.')[-2] + '.tif', depth)
