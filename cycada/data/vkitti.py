import os.path
import sys
import glob

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from .data_loader import register_data_params, register_dataset_obj
from .data_loader import DatasetParams


classes = ['Building', 'Car', 'GuardRail', 'Misc', 'Pole', 'Road', 'Sky', 'Terrain',
           'TrafficLight', 'TrafficSign', 'Tree', 'Truck', 'Van', 'Vegetation']

class2label = {
    'Building': 0,
    'Car': 1,
    'Pole': 2,
    'Road': 3,
    'Sky': 4,
    'Terrain': 5,
    'TrafficLight': 6,
    'TrafficSign': 7,
    'Truck': 8,
    'Vegetation': 9,
    'GuardRail': 10,
    'Misc': 11,
    'Tree': 12,
    'Van': 13
}

label2palette = {
    0: (140, 140, 140),
    1: (200, 200, 200),
    2: (255, 130, 0),
    3: (100, 60, 100),
    4: (90, 200, 255),
    5: (210, 0, 200),
    6: (200, 200, 0),
    7: (255, 255, 0),
    8: (160, 60, 60),
    9: (90, 240, 0),
    10: (255, 100, 255),
    11: (80, 80, 80),
    12: (0, 199, 0),
    13: (230, 208, 202)
}

palette2label = {
    (140, 140, 140): 0,
    (255, 130, 0): 2,
    (100, 60, 100): 3,
    (90, 200, 255): 4,
    (210, 0, 200): 5,
    (200, 200, 0): 6,
    (255, 255, 0): 7,
    (160, 60, 60): 8,
    (90, 240, 0): 9,
    (255, 100, 255): 10,
    (80, 80, 80): 11,
    (0, 199, 0): 12,
    (200, 200, 200): 1, (232, 227, 239): 1, (215, 203, 229): 1, (247, 230, 218): 1, (230, 207, 208): 1,
    (212, 234, 247): 1, (244, 210, 237): 1, (227, 237, 226): 1, (209, 214, 215): 1, (241, 241, 205): 1,
    (224, 217, 244): 1, (206, 244, 234): 1, (239, 221, 223): 1, (221, 248, 212): 1, (203, 224, 202): 1,
    (218, 228, 231): 1, (200, 205, 220): 1, (233, 231, 210): 1, (215, 208, 249): 1, (248, 235, 238): 1,
    (230, 212, 228): 1, (212, 238, 217): 1, (245, 215, 207): 1, (227, 242, 246): 1, (210, 218, 236): 1,
    (242, 245, 225): 1, (224, 222, 214): 1, (207, 249, 204): 1, (239, 225, 243): 1, (221, 202, 233): 1,
    (204, 229, 222): 1, (236, 206, 211): 1, (219, 232, 201): 1, (201, 209, 240): 1, (233, 236, 230): 1,
    (216, 213, 219): 1, (248, 239, 209): 1, (230, 216, 248): 1, (213, 243, 237): 1, (245, 220, 227): 1,
    (228, 246, 216): 1, (210, 223, 206): 1, (242, 200, 245): 1, (239, 230, 213): 1, (222, 207, 203): 1,
    (204, 234, 242): 1, (237, 210, 232): 1, (214, 210, 230): 1, (240, 200, 230): 1, (217, 239, 231): 1,
    (244, 229, 231): 1, (221, 218, 232): 1, (247, 208, 232): 1, (224, 247, 233): 1, (201, 236, 234): 1,
    (228, 226, 234): 1, (204, 215, 235): 1, (231, 205, 235): 1, (208, 244, 236): 1, (235, 233, 237): 1,
    (211, 223, 237): 1, (238, 212, 238): 1, (215, 202, 238): 1, (242, 241, 239): 1, (218, 231, 239): 1,
    (245, 220, 240): 1, (222, 209, 241): 1, (249, 249, 241): 1, (225, 238, 242): 1, (202, 228, 242): 1,
    (229, 217, 243): 1, (206, 206, 244): 1, (232, 246, 244): 1, (209, 235, 245): 1, (236, 225, 245): 1,
    (212, 214, 246): 1, (239, 204, 246): 1, (216, 243, 247): 1, (219, 222, 248): 1, (246, 211, 249): 1,
    (223, 201, 249): 1, (200, 240, 200): 1, (203, 219, 201): 1, (233, 237, 203): 1, (237, 216, 204): 1,
    (214, 205, 205): 1, (240, 245, 205): 1, (217, 234, 206): 1, (244, 224, 206): 1, (236, 201, 241): 1,
    (225, 227, 234): 1, (207, 203, 224): 1, (237, 221, 229): 1, (243, 232, 248): 1, (226, 230, 200): 1,
    (230, 208, 202): 1, (207, 248, 202): 1, (210, 227, 203): 1, (221, 213, 207): 1, (242, 208, 244): 1,
    (231, 209, 205): 1, (221, 209, 216): 1, (210, 210, 227): 1, (200, 210, 238): 1, (239, 211, 249): 1,
    (229, 211, 210): 1, (218, 212, 221): 1, (207, 213, 232): 1, (236, 214, 203): 1, (226, 214, 214): 1,
    (215, 215, 225): 1, (204, 216, 236): 1, (244, 216, 247): 1, (233, 217, 208): 1, (223, 217, 219): 1,
    (202, 218, 241): 1, (241, 219, 202): 1, (230, 220, 213): 1, (220, 220, 224): 1, (209, 221, 235): 1,
    (249, 221, 246): 1, (228, 222, 218): 1, (217, 223, 228): 1, (206, 224, 239): 1, (225, 225, 222): 1,
    (214, 226, 233): 1, (203, 227, 244): 1, (243, 227, 205): 1,
    (230, 208, 202): 13, (207, 248, 202): 13, (210, 227, 203): 13, (227, 237, 226): 13, (206, 244, 234): 13,
    (221, 248, 212): 13, (212, 234, 247): 13, (224, 217, 244): 13, (236, 201, 241): 13, (203, 224, 202): 13,
    (248, 235, 238): 13, (210, 218, 236): 13, (216, 213, 219): 13, (231, 205, 235): 13, (229, 217, 243): 13,
    (232, 246, 244): 13, (219, 222, 248): 13, (247, 213, 242): 13, (212, 218, 230): 13, (246, 224, 200): 13,
    (235, 225, 211): 13
}


def remap_labels_to_palette(arr):
    out = np.zeros([arr.shape[0], arr.shape[1], 3], dtype=np.uint8)
    for label, color in label2palette.items():
        out[arr == label] = color
    return out


def remap_palette_to_labels(arr):
    h, w, c = arr.shape
    out = np.zeros((h, w))
    for rgb, label in palette2label.items():
        mask1 = arr[:, :, 0] == rgb[0]
        mask2 = arr[:, :, 1] == rgb[1]
        mask3 = arr[:, :, 2] == rgb[2]
        mask = mask1 * mask2 * mask3
        out[mask] = label
    return out

@register_data_params('vkitti')
class VKittiParams(DatasetParams):
    num_channels = 3
    image_size = 375
    mean = 0.5
    std = 0.5
    num_cls = 14
    target_transform = None


@register_dataset_obj('vkitti')
class VKitti(data.Dataset):

    def __init__(self, root, split='train', remap_labels=False, transform=None,
                 target_transform=None):
        self.root = root # root = '/media/hpc4_Raid/dsungatullina/submission/vkitti-kitti/vkitti'
        sys.path.append(root)
        self.split = split # no need
        self.remap_labels = remap_labels
        self.ids = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.num_cls = 14
        self.classes = classes

    def collect_ids(self):
        im_dir = os.path.join(self.root, 'vkitti_1.3.1_labels')
        i = 0
        folders = []
        subfolders = []
        for dirpath, dirnames, filenames in os.walk(im_dir):
            if i == 0:
                folders = dirnames
            elif len(dirnames) != 0:
                subfolders.append(dirnames)
            i = i + 1
        assert (len(folders) == len(subfolders))
        ids = []
        for i in range(len(folders)):
            for j in range(len(subfolders[i])):
                imgfolder = os.path.join(self.root, 'vkitti_1.3.1_labels', folders[i], subfolders[i][j])
                for img in glob.glob(os.path.join(imgfolder, '*.png')):
                    ids.append('/'.join(img.split('/')[-3:]))
        ids.sort()
        return ids

    def img_path(self, id):
        return os.path.join(self.root, 'vkitti_1.3.1_rgb', id)

    def label_path(self, id):
        return os.path.join(self.root, 'vkitti_1.3.1_labels', id)

    def __getitem__(self, index):
        id = self.ids[index]
        img = Image.open(self.img_path(id)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = Image.open(self.label_path(id)).convert('L')
        if self.remap_labels:
            target = np.asarray(target)
            target = remap_palette_to_labels(target)
            target = Image.fromarray(np.uint8(target), 'L')
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    cs = VKitti('/x/vkitti')
