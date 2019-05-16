import torch
from torch.autograd import Variable
import util.task as task
from .base_model import BaseModel
from . import network

import torch.nn as nn
import torch.nn.functional as F

#from .FCN8sNet import VGG16_FCN8s
from .FCN8sNet import fcn8s

class FCN8sModel(BaseModel):
    def name(self):
        return 'FCN8s Model'

    def supervised_loss(self, score, target, weight=None, size_average=True):
        n, c, h, w = score.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between score and target
        if h != ht and w != wt:  # upsample labels
            score = F.interpolate(score, size=(ht, wt), mode="bilinear", align_corners=True)

        score = score.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            score, target, weight=weight, size_average=size_average, ignore_index=19
        )
        return loss

    # def supervised_loss(self, score, target, weight=None, size_average=True):
    #     loss_fn_ = torch.nn.NLLLoss2d(weight=weight, size_average=True,
    #                                   ignore_index=255)
    #     loss = loss_fn_(F.log_softmax(self.fake_lab_s), self.lab_s)
    #     return loss

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # self.loss_names = ['lab_s', 'lab_t', 'lab_smooth']
        self.loss_names = ['lab_s', 'lab_t']
        self.visual_names = ['img_s', 'lab_s', 'fake_lab_s', 'img_t', 'lab_t', 'fake_lab_t']
        self.model_names = ['img2task']

        # define the task network
        #self.net_img2task = VGG16_FCN8s()

        self.net_img2task = network.define_Seg(gpu_ids=opt.gpu_ids)

        # if len(self.gpu_ids) > 0:
        #     self.net_img2task.cuda()

        if self.isTrain:
            # define the loss function
            # self.l1loss = torch.nn.L1Loss()
            # self.l2loss = torch.nn.MSELoss()
            # self.task_loss = F.cross_entropy()

            self.optimizer_img2task = torch.optim.SGD(self.net_img2task.parameters(), lr=opt.lr_task, momentum=0.9, weight_decay=0.0005)

            self.optimizers = []
            self.schedulers = []

            self.optimizers.append(self.optimizer_img2task)
            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_source = input['img_source']
        self.img_target = input['img_target']
        if self.isTrain:
            self.lab_source = input['lab_source']
            self.lab_target = input['lab_target']

        # print("self.gpu_ids[0]", self.gpu_ids[0])

        if len(self.gpu_ids) > 0:
            self.img_source = self.img_source.cuda(self.gpu_ids[0], async=True)
            self.img_target = self.img_target.cuda(self.gpu_ids[0], async=True)
            if self.isTrain:
                self.lab_source = self.lab_source.cuda(self.gpu_ids[0], async=True)
                self.lab_target = self.lab_target.cuda(self.gpu_ids[0], async=True)

    def forward(self):
        # print(self.img_source.size(), self.lab_source.size())
        self.img_s = Variable(self.img_source)
        self.fake_lab_s = self.net_img2task(self.img_s)
        self.fake_lab_s = self.fake_lab_s.clamp(min=0.0, max=19.0)
        # print("self.fake_lab_s", self.fake_lab_s.min(), self.fake_lab_s.max())
        # print("self.lab_s", self.lab_source.min(), self.lab_source.max())

    def backward_task(self):

        self.lab_s = Variable(self.lab_source)
        self.loss_lab_s = self.supervised_loss(self.fake_lab_s, self.lab_s)

        # total_loss = self.loss_lab_s + self.loss_lab_smooth
        total_loss = self.loss_lab_s
        total_loss.backward()

    def optimize_parameters(self, epoch_iter):

        self.net_img2task.train()
        self.forward()
        # task network
        self.optimizer_img2task.zero_grad()
        self.backward_task()
        self.optimizer_img2task.step()

    def validation_target(self):

        self.net_img2task.eval()
        self.img_t = Variable(self.img_target, requires_grad=False)
        self.lab_t = Variable(self.lab_target, requires_grad=False)
        self.fake_lab_t = self.net_img2task(self.img_t)
        task_loss = self.supervised_loss(self.fake_lab_t, self.lab_t)

        self.loss_lab_t = task_loss

        self.net_img2task.train()