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

        self.transform_augment_lab = get_transform(opt, True, False)
        self.transform_no_augment_lab = get_transform(opt, False, False)


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

        # if self.opt.crop:
        #     crop_transform = transforms.RandomCrop(self.opt.cropSize)
        #     seed_source = random.randint(0, 2 ** 32)
        #     random.seed(seed_source)
        #     img_source = crop_transform(img_source)
        #     seed_target = random.randint(0, 2 ** 32)
        #     random.seed(seed_target)
        #     img_target = crop_transform(img_target)

        if self.opt.isTrain:
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            if self.opt.dataset_mode == 'paired':
                lab_target_path = self.lab_target_paths[item % self.img_target_size]
            elif self.opt.dataset_mode == 'unpaired':
                lab_target_path = self.lab_target_paths[index]
            else:
                raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

            lab_source = cv2.imread(lab_source_path, cv2.IMREAD_UNCHANGED)
            # tmp_source = np.zeros((lab_source.shape[0], lab_source.shape[1], 3))
            # tmp_source[:, :, 0] = np.copy(lab_source)
            # tmp_source[:, :, 1] = np.copy(lab_source)
            # tmp_source[:, :, 2] = np.copy(lab_source)
            # lab_source = Image.fromarray(tmp_source.astype(int))
            lab_source = torch.from_numpy(lab_source[np.newaxis, :]).float()

            lab_target = cv2.imread(lab_target_path, cv2.IMREAD_UNCHANGED)
            # tmp_target = np.zeros((lab_target.shape[0], lab_target.shape[1], 3))
            # tmp_target[:, :, 0] = np.copy(lab_target)
            # tmp_target[:, :, 1] = np.copy(lab_target)
            # tmp_target[:, :, 2] = np.copy(lab_target)
            # lab_target = Image.fromarray(tmp_target)
            lab_target = torch.from_numpy(lab_target[np.newaxis, :]).float()

            # if self.opt.crop:
            #     random.seed(seed_source)
            #     lab_source = crop_transform(lab_source)
            #     random.seed(seed_target)
            #     lab_target = crop_transform(lab_target)
            #
            # img_source, lab_source, scale = paired_transform(self.opt, img_source, lab_source)
            # img_target, lab_target, scale = paired_transform(self.opt, img_target, lab_target)

            to_tensor = transforms.ToTensor()
            img_source = to_tensor(img_source)
            img_target = to_tensor(img_target)
            # lab_source = to_tensor(lab_source)
            # lab_target = to_tensor(lab_target)

            # img_source = self.transform_augment_img(img_source)
            # lab_source = self.transform_no_augment_lab(lab_source)
            #
            # img_target = self.transform_no_augment_img(img_target)
            # lab_target = self.transform_no_augment_lab(lab_target)

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
        return 'T^2Dataset'


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


def get_transform(opt, augment, isImage):
    transforms_list = []

    if isImage:
        if augment:
            if opt.isTrain:
                transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
        transforms_list += [
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    else:
        transforms_list += [
            transforms.ToTensor(), transforms.Normalize((59.63), (161.4))
        ]

    return transforms.Compose(transforms_list)
