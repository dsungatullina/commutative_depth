import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
from .image_folder import make_dataset
import torchvision.transforms.functional as F

import cv2
import torch
import numpy as np

class CreateDataset(data.Dataset):
    def initialize(self, opt, net_transform=None):
        self.opt = opt

        self.img_source_paths, self.img_source_size = make_dataset(opt.img_source_file)
        self.img_target_paths, self.img_target_size = make_dataset(opt.img_target_file)

        if self.opt.isTrain:
            self.lab_source_paths, self.lab_source_size = make_dataset(opt.lab_source_file)
            assert(self.img_source_size == self.lab_source_size)

        # def get_transform(opt, augment, isImg, net_transform=None)
        self.transform_augment_img = get_transform(opt, True, True, net_transform)
        self.transform_no_augment_lab = get_transform(opt, False, False)

        # print('!!!!!!!!!!!!!!!!!!!!!!')
        # item = 10
        # img_target_path = self.img_target_paths[item % self.img_target_size]
        # print(img_target_path)
        # img_target = Image.open(img_target_path).convert('RGB')
        # if self.opt.resize:
        #     size = (int(self.opt.loadSize.split(',')[0]), int(self.opt.loadSize.split(',')[1]))
        #     print(size)
        #     resize_transform_img = transforms.Resize(size, interpolation=Image.LANCZOS)
        #     img_target = resize_transform_img(img_target)
        # print(img_target.size)
        # img_target.save('/media/hpc4_Raid/dsungatullina/submission/check_target.png')
        # print('!!!!!!!!!!!!!!!!!!!!!!')


    def __getitem__(self, item):
        img_target_path = self.img_target_paths[item % self.img_target_size]

        img_target = Image.open(img_target_path).convert('RGB')
        #img_target.save('/media/hpc4_Raid/dsungatullina/submission/check_target.png')

        if self.opt.resize:
            size = (int(self.opt.loadSize.split(',')[0]), int(self.opt.loadSize.split(',')[1]))
            resize_transform_img = transforms.Resize(size, interpolation=Image.LANCZOS)
            img_target = resize_transform_img(img_target)

        if self.opt.crop:
            crop_transform = transforms.RandomCrop(self.opt.cropSize)
            img_target = crop_transform(img_target)

        if self.opt.isTrain:
            img_source_path = self.img_source_paths[item % self.img_source_size]
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]

            img_source = Image.open(img_source_path).convert('RGB')
            lab_source = Image.open(lab_source_path).convert('L')

            if self.opt.resize:
                img_source = resize_transform_img(img_source)
                resize_transform_lab = transforms.Resize(size, interpolation=Image.NEAREST)
                lab_source = resize_transform_lab(lab_source)

            if self.opt.crop:
                seed_source = random.randint(0, 2 ** 32)
                random.seed(seed_source)
                img_source = crop_transform(img_source)
                lab_source = crop_transform(lab_source)

            img_source, lab_source = paired_transform(self.opt, img_source, lab_source)

            img_target = self.transform_augment_img(img_target)

            img_source = self.transform_augment_img(img_source)
            lab_source = self.transform_no_augment_lab(lab_source)

            return {'img_target': img_target,
                    'img_source': img_source, 'lab_source': lab_source,
                    'img_target_paths': img_target_path,
                    'img_source_paths': img_source_path, 'lab_source_paths': lab_source_path
                    }

        else:
            img_target = self.transform_no_augment_img(img_target)
            return {'img_target': img_target,
                    'img_target_paths': img_target_path
                    }

    def __len__(self):
        return max(self.img_source_size, self.img_target_size)

    def name(self):
        return 'SegDataset'


def dataloader(opt, net_transform):
    datasets = CreateDataset()
    datasets.initialize(opt, net_transform)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=opt.shuffle, num_workers=int(opt.nThreads))
    return dataset

def paired_transform(opt, image, lab):
    if opt.flip:
        n_flip = random.random()
        if n_flip > 0.5:
            image = F.hflip(image)
            lab = F.hflip(lab)
    if opt.rotation:
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BICUBIC)
            lab = F.rotate(lab, degree, Image.BILINEAR)
    return image, lab

def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))

def get_transform(opt, augment, is_img, net_transform=None):
    transforms_list = []
    if is_img: # rgb image
        if augment & opt.isTrain:
            transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
        if net_transform is not None:
            transforms_list += [
                net_transform
            ]
        else:
            transforms_list += [
                transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
    else: # label
        transforms_list += [
            to_tensor_raw
        ]
    return transforms.Compose(transforms_list)