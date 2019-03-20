import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.utils import model_zoo
from torchvision.models import vgg

import functools

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

# FCN 8s
class fcn8s(nn.Module):
    def __init__(self, n_classes=20, learned_billinear=True):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.loss = functools.partial(cross_entropy2d, size_average=False)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        if self.learned_billinear:
            self.upscore2 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore4 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore8 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 16, stride=8, bias=False
            )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(
                    get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                )

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        if self.learned_billinear:
            upscore2 = self.upscore2(score)
            score_pool4c = self.score_pool4(conv4)[
                :, :, 5 : 5 + upscore2.size()[2], 5 : 5 + upscore2.size()[3]
            ]
            upscore_pool4 = self.upscore4(upscore2 + score_pool4c)

            score_pool3c = self.score_pool3(conv3)[
                :, :, 9 : 9 + upscore_pool4.size()[2], 9 : 9 + upscore_pool4.size()[3]
            ]

            out = self.upscore8(score_pool3c + upscore_pool4)[
                :, :, 31 : 31 + x.size()[2], 31 : 31 + x.size()[3]
            ]
            return out.contiguous()

        else:
            score_pool4 = self.score_pool4(conv4)
            score_pool3 = self.score_pool3(conv3)
            score = F.upsample(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.upsample(score, score_pool3.size()[2:])
            score += score_pool3
            out = F.upsample(score, x.size()[2:])

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]


# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision
# from torch import nn
# from torch.autograd import Variable
# from torch.nn import init
# from torch.utils import model_zoo
# from torchvision.models import vgg
#
# #from .models import register_model
#
# def get_upsample_filter(size):
#     """Make a 2D bilinear kernel suitable for upsampling"""
#     factor = (size + 1) // 2
#     if size % 2 == 1:
#         center = factor - 1
#     else:
#         center = factor - 0.5
#     og = np.ogrid[:size, :size]
#     filter = (1 - abs(og[0] - center) / factor) * \
#              (1 - abs(og[1] - center) / factor)
#     return torch.from_numpy(filter).float()
#
#
# class Bilinear(nn.Module):
#
#     def __init__(self, factor, num_channels):
#         super().__init__()
#         self.factor = factor
#         filter = get_upsample_filter(factor * 2)
#         w = torch.zeros(num_channels, num_channels, factor * 2, factor * 2)
#         for i in range(num_channels):
#             w[i, i] = filter
#         self.register_buffer('w', w)
#
#     def forward(self, x):
#         return F.conv_transpose2d(x, Variable(self.w), stride=self.factor)
#
#
# #@register_model('fcn8s')
# class VGG16_FCN8s(nn.Module):
#
#     transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]),
#         ])
#
#     def __init__(self, num_cls=19, pretrained=True, weights_init=None,
#             output_last_ft=False):
#         super().__init__()
#         self.output_last_ft = output_last_ft
#         self.vgg = make_layers(vgg.cfg['D'])
#         self.vgg_head = nn.Sequential(
#             nn.Conv2d(512, 4096, 7),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(4096, 4096, 1),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(4096, num_cls, 1)
#             )
#         self.upscore2 = self.upscore_pool4 = Bilinear(2, num_cls)
#         self.upscore8 = Bilinear(8, num_cls)
#         self.score_pool4 = nn.Conv2d(512, num_cls, 1)
#         for param in self.score_pool4.parameters():
#             init.constant(param, 0)
#         self.score_pool3 = nn.Conv2d(256, num_cls, 1)
#         for param in self.score_pool3.parameters():
#             init.constant(param, 0)
#
#         if pretrained:
#             if weights_init is not None:
#                 self.load_weights(torch.load(weights_init))
#             else:
#                 self.load_base_weights()
#
#     def load_base_vgg(self, weights_state_dict):
#         vgg_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg.')
#         self.vgg.load_state_dict(vgg_state_dict)
#
#     def load_vgg_head(self, weights_state_dict):
#         vgg_head_state_dict = self.get_dict_by_prefix(weights_state_dict, 'vgg_head.')
#         self.vgg_head.load_state_dict(vgg_head_state_dict)
#
#     def get_dict_by_prefix(self, weights_state_dict, prefix):
#         return {k[len(prefix):]: v
#                 for k,v in weights_state_dict.items()
#                 if k.startswith(prefix)}
#
#
#     def load_weights(self, weights_state_dict):
#         self.load_base_vgg(weights_state_dict)
#         self.load_vgg_head(weights_state_dict)
#
#     def split_vgg_head(self):
#         self.classifier = list(self.vgg_head.children())[-1]
#         self.vgg_head_feat = nn.Sequential(*list(self.vgg_head.children())[:-1])
#
#
#     def forward(self, x):
#         input = x
#         x = F.pad(x, (99, 99, 99, 99), mode='constant', value=0)
#         intermediates = {}
#         fts_to_save = {16: 'pool3', 23: 'pool4'}
#         for i, module in enumerate(self.vgg):
#             x = module(x)
#             if i in fts_to_save:
#                 intermediates[fts_to_save[i]] = x
#
#         ft_to_save = 5 # Dropout before classifier
#         last_ft = {}
#         for i, module in enumerate(self.vgg_head):
#             x = module(x)
#             if i == ft_to_save:
#                 last_ft = x
#
#         _, _, h, w = x.size()
#         upscore2 = self.upscore2(x)
#         pool4 = intermediates['pool4']
#         score_pool4 = self.score_pool4(0.01 * pool4)
#         score_pool4c = _crop(score_pool4, upscore2, offset=5)
#         fuse_pool4 = upscore2 + score_pool4c
#         upscore_pool4 = self.upscore_pool4(fuse_pool4)
#         pool3 = intermediates['pool3']
#         score_pool3 = self.score_pool3(0.0001 * pool3)
#         score_pool3c = _crop(score_pool3, upscore_pool4, offset=9)
#         fuse_pool3 = upscore_pool4 + score_pool3c
#         upscore8 = self.upscore8(fuse_pool3)
#         score = _crop(upscore8, input, offset=31)
#         if self.output_last_ft:
#             return score, last_ft
#         else:
#             return score
#
#
#     def load_base_weights(self):
#         """This is complicated because we converted the base model to be fully
#         convolutional, so some surgery needs to happen here."""
#         base_state_dict = model_zoo.load_url(vgg.model_urls['vgg16'])
#         vgg_state_dict = {k[len('features.'):]: v
#                           for k, v in base_state_dict.items()
#                           if k.startswith('features.')}
#         self.vgg.load_state_dict(vgg_state_dict)
#         vgg_head_params = self.vgg_head.parameters()
#         for k, v in base_state_dict.items():
#             if not k.startswith('classifier.'):
#                 continue
#             if k.startswith('classifier.6.'):
#                 # skip final classifier output
#                 continue
#             vgg_head_param = next(vgg_head_params)
#             vgg_head_param.data = v.view(vgg_head_param.size())
#
#
#
# # class VGG16_FCN8s_caffe(VGG16_FCN8s):
# #
# #     transform = torchvision.transforms.Compose([
# #         torchvision.transforms.ToTensor(),
# #         torchvision.transforms.Normalize(
# #             mean=[0.485, 0.458, 0.408],
# #             std=[0.00392156862745098] * 3),
# #         torchvision.transforms.Lambda(
# #             lambda x: torch.stack(torch.unbind(x, 1)[::-1], 1))
# #         ])
# #
# #     def load_base_weights(self):
# #         base_state_dict = model_zoo.load_url('https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth')
# #         vgg_state_dict = {k[len('features.'):]: v
# #                           for k, v in base_state_dict.items()
# #                           if k.startswith('features.')}
# #         self.vgg.load_state_dict(vgg_state_dict)
# #         vgg_head_params = self.vgg_head.parameters()
# #         for k, v in base_state_dict.items():
# #             if not k.startswith('classifier.'):
# #                 continue
# #             if k.startswith('classifier.6.'):
# #                 # skip final classifier output
# #                 continue
# #             vgg_head_param = next(vgg_head_params)
# #             vgg_head_param.data = v.view(vgg_head_param.size())
# #
# # class Discriminator(nn.Module):
# #     def __init__(self, input_dim=4096, output_dim=2, pretrained=False, weights_init=''):
# #         super().__init__()
# #         dim1 = 1024 if input_dim==4096 else 512
# #         dim2 = int(dim1/2)
# #         self.D = nn.Sequential(
# #             nn.Conv2d(input_dim, dim1, 1),
# #             nn.Dropout2d(p=0.5),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(dim1, dim2, 1),
# #             nn.Dropout2d(p=0.5),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(dim2, output_dim, 1)
# #             )
# #
# #         if pretrained and weights_init is not None:
# #             self.load_weights(weights_init)
# #
# #     def forward(self, x):
# #         d_score = self.D(x)
# #         return d_score
# #
# #     def load_weights(self, weights):
# #         print('Loading discriminator weights')
# #         self.load_state_dict(torch.load(weights))
# #
# #
# #
# # class Transform_Module(nn.Module):
# #     def __init__(self, input_dim=4096):
# #         super().__init__()
# #         self.transform = nn.Sequential(
# #             nn.Conv2d(input_dim, input_dim, 1),
# #             nn.ReLU(inplace=True),
# #             #nn.Conv2d(input_dim, input_dim, 1),
# #             #nn.ReLU(inplace=True),
# #             )
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 init_eye(m.weight)
# #                 m.bias.data.zero_()
# #
# #     def forward(self, x):
# #         t_x = self.transform(x)
# #         return t_x
# #
# #
# # def init_eye(tensor):
# #     if isinstance(tensor, Variable):
# #         init_eye(tensor.data)
# #         return tensor
# #     return tensor.copy_(torch.eye(tensor.size(0), tensor.size(1)))
#
#
# def _crop(input, shape, offset=0):
#     _, _, h, w = shape.size()
#     return input[:, :, offset:offset + h, offset:offset + w].contiguous()
#
#
# def make_layers(cfg, batch_norm=False):
#     """This is almost verbatim from torchvision.models.vgg, except that the
#     MaxPool2d modules are configured with ceil_mode=True.
#     """
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             modules = [conv2d, nn.ReLU(inplace=True)]
#             if batch_norm:
#                 modules.insert(1, nn.BatchNorm2d(v))
#             layers.extend(modules)
#             in_channels = v
#     return nn.Sequential(*layers)
