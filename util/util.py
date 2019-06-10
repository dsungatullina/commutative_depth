import numpy as np
import os
import imageio
import matplotlib.cm as mpl

from torch.nn.parameter import Parameter

def safe_load_state_dict(net, state_dict):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. Any params in :attr:`state_dict`
    that do not match the keys returned by :attr:`net`'s :func:`state_dict()`
    method or have differing sizes are skipped.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
    """
    own_state = net.state_dict()
    skipped = []
    for name, param in state_dict.items():
        if name not in own_state:
            skipped.append(name)
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if own_state[name].size() != param.size():
            skipped.append(name)
            continue
        own_state[name].copy_(param)

    if skipped:
        logging.info('Skipped loading some parameters: {}'.format(skipped))

# convert a tensor into a numpy array
def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3: # grayscale
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes
        return image_numpy.astype(imtype)
    elif image_tensor.size(1) == 3: # rgb
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * bytes
        return image_numpy.astype(imtype)
    else: # segmentation maps
        image_numpy = image_tensor[0].argmax(dim=0).cpu().numpy()
        image_numpy = remap_labels_to_palette(image_numpy.astype(np.uint8))
        return image_numpy

# seg

# vkitti
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

def renorm_rgb(arr):
    arr[:, :, 0] = arr[:, :, 0] * 0.229 + 0.485
    arr[:, :, 1] = arr[:, :, 1] * 0.224 + 0.456
    arr[:, :, 2] = arr[:, :, 2] * 0.225 + 0.406
    return arr

# convert a tensor into a numpy array
def tensor2im_(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3: # grayscale
        image_numpy = image_tensor[0].cpu().numpy()
        image_numpy = remap_labels_to_palette(image_numpy.astype(imtype))
        return image_numpy
    elif image_tensor.size(1) == 3: # rgb
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_numpy = renorm_rgb(image_numpy)
        image_numpy = image_numpy * bytes
        return image_numpy.astype(imtype)
    else: # segmentation maps
        image_numpy = image_tensor[0].argmax(dim=0).cpu().numpy()
        image_numpy = remap_labels_to_palette(image_numpy.astype(imtype))
        return image_numpy

def depth2colormap(image, colormap='jet'):
    """
    :param image:numpy array from [0,1]
    :param colormap: string (name of the colormap)
    :return:
    """
    cm = mpl.get_cmap(colormap)
    return cm(image)


# convert a tensor into a numpy array
def tensor2im_2(image_tensor, bytes=255.0, imtype=np.uint8):
    if image_tensor.dim() == 3: # gt labels
        image_numpy = image_tensor[0].cpu().numpy()
        image_numpy = remap_labels_to_palette(image_numpy.astype(imtype))
        return image_numpy
    elif image_tensor.size(1) == 1:  # depth
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (image_numpy[0, :, :] + 1.0) / 2.0
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0
        image_numpy = depth2colormap(image_numpy)
        image_numpy = image_numpy * bytes
        return image_numpy.astype(imtype)
    elif image_tensor.size(1) == 3: # rgb
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0
        image_numpy = image_numpy * bytes
        return image_numpy.astype(imtype)
    else: # segmentation maps
        image_numpy = image_tensor[0].argmax(dim=0).cpu().numpy()
        image_numpy = remap_labels_to_palette(image_numpy.astype(imtype))
        return image_numpy

def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
