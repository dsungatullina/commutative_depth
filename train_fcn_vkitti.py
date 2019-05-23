import logging
import os.path
from collections import deque

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable

from cycada.data.data_loader import get_fcn_dataset as get_dataset
from cycada.models import get_model
from cycada.models.models import models
from cycada.transforms import augment_collate
from cycada.util import config_logging
from cycada.util import to_tensor_raw
from cycada.tools.util import make_variable

def to_tensor_raw(im):
    return torch.from_numpy(np.array(im, np.int64, copy=False))

def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])

def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, reduction='mean',
            ignore_index=255)
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss


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

def remap_labels_to_palette(arr):
    out = np.zeros([arr.shape[0], arr.shape[1], 3], dtype=np.uint8)
    for label, color in label2palette.items():
        out[arr == label] = color
    return out

@click.command()
@click.argument('output')
@click.option('--dataset', required=True, multiple=True)
@click.option('--datadir', default="", type=click.Path(exists=True))
@click.option('--batch_size', '-b', default=1)
@click.option('--lr', '-l', default=0.001)
@click.option('--step', type=int)
@click.option('--iterations', '-i', default=100000)
@click.option('--momentum', '-m', default=0.9)
@click.option('--snapshot', '-s', default=5000)
@click.option('--downscale', type=int, default=None)
@click.option('--augmentation/--no-augmentation', default=False)
@click.option('--fyu/--torch', default=False)
@click.option('--crop_size', default=None)
@click.option('--weights', type=click.Path(exists=True))
@click.option('--model', default='fcn8s', type=click.Choice(models.keys()))
@click.option('--num_cls', default=19, type=int)
@click.option('--gpu', default='0')
def main(output, dataset, datadir, batch_size, lr, step, iterations, 
        momentum, snapshot, downscale, augmentation, fyu, crop_size, 
        weights, model, gpu, num_cls):
    if weights is not None:
        raise RuntimeError("weights don't work because eric is bad at coding")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config_logging()
    logdir = 'runs/{:s}/{:s}'.format(model, '-'.join(dataset))
    writer = SummaryWriter(log_dir=logdir)
    #net = get_model(model, num_cls=num_cls)
    net = get_model(model, num_cls=num_cls, finetune=True)
    net.cuda()
    transform = []
    target_transform = []
    # if downscale is not None:
    #     transform.append(torchvision.transforms.Resize(1024 // downscale))
    #     target_transform.append(
    #         torchvision.transforms.Resize(1024 // downscale,
    #                                      interpolation=Image.NEAREST))
    transform.extend([
#        torchvision.transforms.Resize((192, 640), interpolation=Image.LANCZOS),
        net.transform
        ])
    target_transform.extend([
#        torchvision.transforms.Resize((192, 640), interpolation=Image.NEAREST),
        to_tensor_raw
        ])
    transform = torchvision.transforms.Compose(transform)
    target_transform = torchvision.transforms.Compose(target_transform)

    datasets = [get_dataset(name, datadir, transform=transform,
                            target_transform=target_transform)
                for name in dataset]
    if weights is not None:
        weights = np.loadtxt(weights)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                          weight_decay=0.0005)

    if augmentation:
        collate_fn = lambda batch: augment_collate(batch, crop=crop_size, flip=True)
    else:
        collate_fn = torch.utils.data.dataloader.default_collate
    loaders = [torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2,
                                           collate_fn=collate_fn,
                                           pin_memory=True)
               for dataset in datasets]
    iteration = 0
    losses = deque(maxlen=10)
    for im, label in roundrobin_infinite(*loaders):
        # Clear out gradients
        opt.zero_grad()
        
        # load data/label
        im = make_variable(im, requires_grad=False)
        label = make_variable(label, requires_grad=False)

        # forward pass and compute loss
        preds = net(im)
        loss = supervised_loss(preds, label)
        
        # backward pass
        loss.backward()
        losses.append(loss.data.item())
        
        # step gradients
        opt.step()

        # log results
        if iteration % 1 == 0:
            logging.info('Iteration {}:\t{}'
                            .format(iteration, np.mean(losses)))
            writer.add_scalar('loss', np.mean(losses), iteration)

            print('loss: ', loss.data.item())

            outdir = '/'.join(output.split('/')[:-1])

            if not os.path.exists(outdir + '/images'):
                os.makedirs(outdir + '/images')

            for i in range(label.size(0)):

                tmp1 = im.data[i].cpu().numpy().transpose(1,2,0)
                tmp1[:, :, 0] = 255 * (tmp1[:, :, 0] * 0.229 + 0.485)
                tmp1[:, :, 1] = 255 * (tmp1[:, :, 1] * 0.224 + 0.456)
                tmp1[:, :, 2] = 255 * (tmp1[:, :, 2] * 0.225 + 0.406)
                tmp1 = Image.fromarray(tmp1.astype(np.uint8)).convert('RGB')
                tmp1.save(outdir + '/images/' + '{}_img_#{}.jpg'.format(iteration, i))

                tmp1 = label.data[i].cpu().numpy()
                tmp1 = tmp1.astype(np.uint8)
                tmp1 = Image.fromarray(remap_labels_to_palette(tmp1)).convert('RGB')
                tmp1.save(outdir + '/images/' + '{}_label_#{}.jpg'.format(iteration, i))

                tmp1 = preds.data[i].argmax(dim=0).cpu().numpy()
                tmp1 = tmp1.astype(np.uint8)
                tmp1 = Image.fromarray(remap_labels_to_palette(tmp1)).convert('RGB')
                tmp1.save(outdir + '/images/' + '{}_pred_#{}.jpg'.format(iteration, i))

        iteration += 1
        if step is not None and iteration % step == 0:
            logging.info('Decreasing learning rate by 0.1.')
            step_lr(optimizer, 0.1)
        if iteration % snapshot == 0:
            torch.save(net.state_dict(),
                        '{}-iter{}.pth'.format(output, iteration))
        if iteration >= iterations:
            logging.info('Optimization complete.')
            break
                

if __name__ == '__main__':
    main()
