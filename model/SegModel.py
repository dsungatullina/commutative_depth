import torch
from torch.autograd import Variable
import util.task as task
from .base_model import BaseModel
from . import network

import torch.nn.functional as F
from collections import OrderedDict
from .models import get_model
from util import util

class SegNetModel(BaseModel):
    def name(self):
        return 'SegNet Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['lab_s']
        self.visual_names = ['img_s', 'lab_s', 'lab_s_g']
        self.model_names = ['img2seg']

        # define the task network
        self.net_img2seg = get_model(opt.seg_model_name, num_cls=opt.num_cls, finetune=True)

        if self.isTrain:

            self.optimizer_img2seg = torch.optim.SGD(self.net_img2seg.parameters(), lr=opt.lr_seg,
                                                     momentum=opt.momentum_seg)

            self.optimizers = []
            self.schedulers = []

            self.optimizers.append(self.optimizer_img2seg)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        # TODO continue train
        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.which_epoch)

    # return visualization images
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    visual_ret[name] = util.tensor2im_(value[-1].data)
                else:
                    visual_ret[name] = util.tensor2im_(value.data)
        return visual_ret

    def supervised_loss(self, score, label, weights=None):
        loss_fn_ = torch.nn.NLLLoss(weight=weights, reduction='mean', ignore_index=255)
        loss = loss_fn_(F.log_softmax(score, dim=1), label)
        return loss

    def set_input(self, input):
        self.input = input
        self.img_target = input['img_target']
        if self.isTrain:
            self.img_source = input['img_source']
            self.lab_source = input['lab_source']

        if len(self.gpu_ids) > 0:
            self.img_target = self.img_target.cuda(self.gpu_ids[0], async=True)
            if self.isTrain:
                self.img_source = self.img_source.cuda(self.gpu_ids[0], async=True)
                self.lab_source = self.lab_source.cuda(self.gpu_ids[0], async=True)

    def forward(self):
        self.img_t = Variable(self.img_target)
        self.img_s = Variable(self.img_source)
        self.lab_s = Variable(self.lab_source)

    # def foreward_G_basic(self, net_G, img_s, img_t):
    #
    #     img = torch.cat([img_s, img_t], 0)
    #     fake = net_G(img)
    #
    #     size = len(fake)
    #
    #     f_s, f_t = fake[0].chunk(2)
    #     img_fake = fake[1:]
    #
    #     img_s_fake = []
    #     img_t_fake = []
    #
    #     for img_fake_i in img_fake:
    #         img_s, img_t = img_fake_i.chunk(2)
    #         img_s_fake.append(img_s)
    #         img_t_fake.append(img_t)
    #
    #     return img_s_fake, img_t_fake, f_s, f_t, size

    def forward_G_basic(self, net_G, img_s, img_t):
        labels = net_G(img_s)
        return labels

    def backward_seg(self):
        self.lab_s_g = self.forward_G_basic(self.net_img2seg, self.img_s, self.img_t)
        self.loss_lab_s = self.supervised_loss(self.lab_s_g, self.lab_s)

        self.loss_lab_s.backward()

    def optimize_parameters(self, epoch_iter):

        self.forward()
        # task network
        self.optimizer_img2seg.zero_grad()
        self.backward_seg()
        self.optimizer_img2seg.step()


