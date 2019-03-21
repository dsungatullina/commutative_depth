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
    def initialize(self, opt):
        self.opt = opt

        self.img_source_paths, self.img_source_size = make_dataset(opt.img_source_file)
        self.img_target_paths, self.img_target_size = make_dataset(opt.img_target_file)

        if self.opt.isTrain:
            self.lab_source_paths, self.lab_source_size = make_dataset(opt.lab_source_file)
            # for visual results, not for training
            self.lab_target_paths, self.lab_target_size = make_dataset(opt.lab_target_file)

        self.transform_augment_img = get_transform(opt, True, True)
        self.transform_no_augment_img = get_transform(opt, False, True)

        self.transform_no_augment_lab_synt = get_transform(opt, False, False, True)
        self.transform_no_augment_lab_real = get_transform(opt, False, False, False)


    def __getitem__(self, item):
        index = random.randint(0, self.img_target_size - 1)
        img_source_path = self.img_source_paths[item % self.img_source_size]
        if self.opt.dataset_mode == 'paired':
            img_target_path = self.img_target_paths[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            img_target_path = self.img_target_paths[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_source = Image.open(img_source_path).convert('RGB')
        img_target = Image.open(img_target_path).convert('RGB')

        if self.opt.crop:
            crop_transform = transforms.RandomCrop(self.opt.cropSize)
            seed_source = random.randint(0, 2 ** 32)
            random.seed(seed_source)
            img_source = crop_transform(img_source)
            seed_target = random.randint(0, 2 ** 32)
            random.seed(seed_target)
            img_target = crop_transform(img_target)

        if self.opt.isTrain:
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            if self.opt.dataset_mode == 'paired':
                lab_target_path = self.lab_target_paths[item % self.img_target_size]
            elif self.opt.dataset_mode == 'unpaired':
                lab_target_path = self.lab_target_paths[index]
            else:
                raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

            lab_source = cv2.imread(lab_source_path, cv2.IMREAD_UNCHANGED)
            lab_source[lab_source > self.opt.maxRealDepth] = self.opt.maxRealDepth
            lab_source = (lab_source / self.opt.maxRealDepth) * 255
            # print(lab_source.shape)
            # lab_source = lab_source[np.newaxis, :]
            lab_source = Image.fromarray(lab_source.astype(np.uint8))
            lab_source = lab_source.convert('L')

            lab_target = cv2.imread(lab_target_path, cv2.IMREAD_UNCHANGED)
            lab_target[lab_target > self.opt.maxRealDepth] = self.opt.maxRealDepth
            lab_target = (lab_target / self.opt.maxRealDepth) * 255
            # lab_target = lab_target[np.newaxis, :]
            lab_target = Image.fromarray(lab_target.astype(np.uint8))
            lab_target = lab_target.convert('L')

            if self.opt.crop:
                random.seed(seed_source)
                lab_source = crop_transform(lab_source)
                random.seed(seed_target)
                lab_target = crop_transform(lab_target)

            img_source, lab_source, scale = paired_transform(self.opt, img_source, lab_source)
            img_target, lab_target, scale = paired_transform(self.opt, img_target, lab_target)

            img_source = self.transform_augment_img(img_source)
            lab_source = self.transform_no_augment_lab_synt(lab_source)

            img_target = self.transform_no_augment_img(img_target)
            lab_target = self.transform_no_augment_lab_real(lab_target)

            # print(lab_source.size())

            return {'img_source': img_source, 'img_target': img_target,
                    'lab_source': lab_source, 'lab_target': lab_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path, 'lab_target_paths': lab_target_path
                    }

        else:
            img_source = self.transform_augment_img(img_source)
            img_target = self.transform_no_augment_img(img_target)
            return {'img_source': img_source, 'img_target': img_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    }

    def __len__(self):
        return max(self.img_source_size, self.img_target_size)

    def name(self):
        return 'Dataset'


def dataloader(opt):
    datasets = CreateDataset()
    datasets.initialize(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batchSize, shuffle=opt.shuffle, num_workers=int(opt.nThreads))
    return dataset

def paired_transform(opt, image, depth):
    scale_rate = 1.0

    if opt.flip:
        n_flip = random.random()
        if n_flip > 0.5:
            image = F.hflip(image)
            depth = F.hflip(depth)

    if opt.rotation:
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BICUBIC)
            depth = F.rotate(depth, degree, Image.BILINEAR)

    return image, depth, scale_rate


def get_transform(opt, augment, isRGB, isSynt=True):
    transforms_list = []

    if isRGB:
        if augment & opt.isTrain:
            transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
        transforms_list += [
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    else:
        if isSynt: # Synthia depth mean and std in meters, [0,1]
            transforms_list += [
                # transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                transforms.ToTensor(), transforms.Normalize([opt.mean_depth_synt], [opt.std_depth_synt])
            ]
        else: # Cityscapes depth mean and std in meters, [0,1]
            transforms_list += [
                transforms.ToTensor(), transforms.Normalize([opt.mean_depth_real], [opt.std_depth_real])
            ]

    return transforms.Compose(transforms_list)
