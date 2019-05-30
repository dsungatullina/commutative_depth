import torch
import itertools
import util.task as task
from torch.autograd import Variable
from .base_model import BaseModel
from util.image_pool import ImagePool
from . import network
from . import networks_cg

import torchvision
import torch.nn.functional as F
from collections import OrderedDict
from .models import get_model
from util import util


class ComSegModel(BaseModel):
    def name(self):
        return 'ComSeg Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # self.loss_names = ['G_R2S', 'G_S2R', 'D_S', 'D_R', 'cycle_S', 'cycle_R', 'com_S', 'com_R', 'l1_DS']
        self.loss_names = ['G_R2S', 'G_S2R', 'D_S', 'D_R', 'cycle_S', 'cycle_R', 'lab_s']
        self.visual_names_S = ['img_s', 'fake_r', 'lab_s', 'lab_s_g', 'lab_fake_r_g']
        self.visual_names_R = ['img_r', 'fake_s',          'lab_r_g', 'lab_fake_s_g']
        self.visual_names_Others = ['rec_r', 'rec_s', 'idt_r', 'idt_s']
        if self.isTrain:
            self.model_names = ['img2seg', 'S2R', 'R2S', 'D_S', 'D_R']
        else:
            self.model_names = ['img2seg', 'S2R', 'R2S']

        self.visual_names = self.visual_names_S + self.visual_names_R

        # define the task network
        self.net_img2seg = get_model(opt.seg_model_name, num_cls=opt.num_cls, finetune=True)

        #define the S->R and R->S networks
        self.net_S2R = networks_cg.define_G(opt.image_nc, opt.image_nc, opt.ngf, 'resnet_9blocks', norm='instance',
                                                 gpu_ids=self.gpu_ids)
        self.net_R2S = networks_cg.define_G(opt.image_nc, opt.image_nc, opt.ngf, 'resnet_9blocks', norm='instance',
                                                 gpu_ids=self.gpu_ids)

        # self.renorm = torchvision.transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225])

        # self.norm = torchvision.transforms.Normalize(
        #     mean=[0.5, 0.5, 0.5],
        #     std=[0.5, 0.5, 0.5])

        if opt.init_models:
            # load seg net
            self.net_img2seg.load_state_dict(torch.load(opt.init_seg_netG_filename))

            # load S2R net
            if isinstance(self.net_S2R, torch.nn.DataParallel):
                self.net_S2R = self.net_S2R.module
            state_dict = torch.load(opt.init_S2R_netG_filename, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            self.net_S2R.load_state_dict(state_dict)

            # load R2S net
            if isinstance(self.net_R2S, torch.nn.DataParallel):
                self.net_R2S = self.net_R2S.module
            state_dict = torch.load(opt.init_R2S_netG_filename, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            self.net_R2S.load_state_dict(state_dict)

            print('translators and the task net are loaded.')

        if self.isTrain:
            # define discriminators
            self.net_D_S = networks_cg.define_D(opt.image_nc, 64, 'basic', 3, 'instance', 'normal', 0.02, self.gpu_ids)
            self.net_D_R = networks_cg.define_D(opt.image_nc, 64, 'basic', 3, 'instance', 'normal', 0.02, self.gpu_ids)
            if opt.init_models:
                if isinstance(self.net_D_S, torch.nn.DataParallel):
                    self.net_D_S = self.net_D_S.module
                state_dict = torch.load(opt.init_R2S_netD_filename, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                self.net_D_S.load_state_dict(state_dict)

                if isinstance(self.net_D_R, torch.nn.DataParallel):
                    self.net_D_R = self.net_D_R.module
                state_dict = torch.load(opt.init_S2R_netD_filename, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                self.net_D_R.load_state_dict(state_dict)

                print('discriminator nets are loaded.')

            # create image buffers to store previously generated images
            self.fake_img_s_pool = ImagePool(opt.batchSize)
            self.fake_img_r_pool = ImagePool(opt.batchSize)

            # define loss functions
                # cyclegan losses
            self.criterionGAN = networks_cg.GANLoss(self.opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # define optimizers
            self.optimizer_T2 = torch.optim.Adam(
                [{'params': self.net_img2seg.parameters(), 'lr': opt.lr_seg, 'betas': (opt.momentum_seg, 0.999)},
                 {'params': itertools.chain(self.net_S2R.parameters(), self.net_R2S.parameters()),
                  'lr': opt.lr_trans, 'betas': (0.5, 0.9)}])

            # # define optimizers
            # self.optimizer_T2 = torch.optim.SGD([
            #     {'params': itertools.chain(self.net_S2R.parameters(), self.net_R2S.parameters()), 'lr': opt.lr_task},
            #     {'params': self.net_img2seg.parameters()}],
            #     lr=opt.lr_seg, momentum=opt.momentum_seg)

            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.net_D_S.parameters(), self.net_D_R.parameters()),
                lr=opt.lr_trans, betas=(0.5, 0.9))

            self.optimizers = []
            self.schedulers = []

            self.optimizers.append(self.optimizer_T2)
            self.optimizers.append(self.optimizer_D)

            for optimizer in self.optimizers:
                self.schedulers.append(network.get_scheduler(optimizer, opt))

        # TODO fix
        # if not self.isTrain or opt.continue_train:
        #     self.load_networks(opt.which_epoch)

    # return visualization images
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                if isinstance(value, list):
                    visual_ret[name] = util.tensor2im_2(value[-1].data)
                else:
                    visual_ret[name] = util.tensor2im_2(value.data)
        return visual_ret

    def supervised_loss(self, score, label, weights=None):
        loss_fn_ = torch.nn.NLLLoss(weight=weights, reduction='mean', ignore_index=255)
        loss = loss_fn_(F.log_softmax(score, dim=1), label)
        return loss

    def renorm(self, tensor):
        tensor = (tensor + 1.0) / 2.0
        tensor[:, 0, :, :] = (tensor[:, 0, :, :] - 0.485) / 0.229
        tensor[:, 1, :, :] = (tensor[:, 1, :, :] - 0.456) / 0.224
        tensor[:, 2, :, :] = (tensor[:, 2, :, :] - 0.406) / 0.225
        return tensor

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
        self.img_s = Variable(self.img_source)
        self.img_r = Variable(self.img_target)
        self.lab_s = Variable(self.lab_source)

        self.fake_s = self.net_R2S(self.img_r)
        self.fake_r = self.net_S2R(self.img_s)
        # cycle
        self.rec_s = self.net_R2S(self.fake_r)
        self.rec_r = self.net_S2R(self.fake_s)

        # identity term
        self.idt_s = self.net_R2S(self.img_s)
        self.idt_r = self.net_S2R(self.img_r)

        # seg
        self.lab_s_g = self.net_img2seg(self.renorm(self.img_s))
        self.lab_r_g = self.net_img2seg(self.renorm(self.img_r))
        self.lab_fake_r_g = self.net_img2seg(self.renorm(self.fake_r))
        self.lab_fake_s_g = self.net_img2seg(self.renorm(self.fake_s))

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_S(self):
        fake_s = self.fake_img_s_pool.query(self.fake_s)
        self.loss_D_S = self.backward_D_basic(self.net_D_S, self.img_s, fake_s)

    def backward_D_R(self):
        fake_r = self.fake_img_r_pool.query(self.fake_r)
        self.loss_D_R = self.backward_D_basic(self.net_D_R, self.img_r, fake_r)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_S = self.opt.lambda_S    # syn
        lambda_R = self.opt.lambda_R    # real
        # GAN loss
        self.loss_G_R2S = self.criterionGAN(self.net_D_S(self.fake_s), True)
        self.loss_G_S2R = self.criterionGAN(self.net_D_R(self.fake_r), True)
        # Forward cycle loss
        self.loss_cycle_S = self.criterionCycle(self.rec_s, self.img_s) * lambda_S
        # Backward cycle loss
        self.loss_cycle_R = self.criterionCycle(self.rec_r, self.img_r) * lambda_R
        # Identity loss
        if lambda_idt > 0:
            self.loss_idt_S = self.criterionIdt(self.idt_s, self.img_s) * lambda_S * lambda_idt
            self.loss_idt_R = self.criterionIdt(self.idt_r, self.img_r) * lambda_R * lambda_idt
        else:
            self.loss_idt_S = 0
            self.loss_idt_R = 0

        # total CycleGAN loss
        self.loss_G = self.loss_G_R2S + self.loss_G_S2R + self.loss_cycle_S \
                      + self.loss_cycle_R + self.loss_idt_S + self.loss_idt_R

        # segmentation loss
        self.loss_lab_s = self.supervised_loss(self.lab_s_g, self.lab_s)

        # total loss
        total_loss = self.loss_G + self.loss_lab_s
        total_loss.backward()

#         ####
#         # tmp1 = self.gen_depth_s[4].clone()
#         # self.loss_DS2R_DS = self.crtiterionCom_DS2R_DS(self.gen_depth_fake_r[4], tmp1.detach())
#         # tmp2 = self.gen_depth_fake_r[4].clone()
#         # self.loss_DS_DS2R = self.crtiterionCom_DR2S_DR(self.gen_depth_s[4], tmp2.detach())
#         # self.loss_com_S =  0.5 * (self.loss_DS2R_DS + self.loss_DS_DS2R) * self.opt.lambda_com_S
#         #
#         # tmp3 = self.gen_depth_r[4].clone()
#         # self.loss_DR2S_DR = self.crtiterionCom_DS_DS2R(self.gen_depth_fake_s[4], tmp3.detach())
#         # tmp4 = self.gen_depth_fake_s[4].clone()
#         # self.loss_DR_DR2S =  self.crtiterionCom_DR_DR2S(self.gen_depth_r[4], tmp4.detach())
#         # self.loss_com_R = 0.5 * (self.loss_DR2S_DR + self.loss_DR_DR2S) * self.opt.lambda_com_R
#         #
#         # self.loss_D_DS = self.crtiterionCom_D_DS(self.gen_depth_s[4], self.depth_s.detach())
#         # self.loss_l1_DS = self.loss_D_DS * self.opt.lambda_l1_DS
#
#         if self.opt.com_loss == 'usual':
#             # com S
#             loss_com_S = torch.mean(torch.abs(self.gen_depth_fake_r[4] - self.gen_depth_s[4]))
#             self.loss_com_S = loss_com_S * self.opt.lambda_com_S
#             # com R
#             loss_com_R = torch.mean(torch.abs(self.gen_depth_fake_s[4] - self.gen_depth_r[4]))
#             self.loss_com_R = loss_com_R * self.opt.lambda_com_R
#         elif self.opt.com_loss == 'pyramid':
#             # com S
#             loss_com_S = 0.0
#             for (i_gen_depth_fake_r, i_gen_depth_s) in zip(self.gen_depth_fake_r, self.gen_depth_s):
#                 loss_com_S = loss_com_S + torch.mean(torch.abs(i_gen_depth_fake_r - i_gen_depth_s))
#             self.loss_com_S = loss_com_S * self.opt.lambda_com_S
#             # com R
#             loss_com_R = 0.0
#             for (i_gen_depth_fake_s, i_gen_depth_r) in zip(self.gen_depth_fake_s, self.gen_depth_r):
#                 loss_com_R = loss_com_R + torch.mean(torch.abs(i_gen_depth_fake_s - i_gen_depth_r))
#             self.loss_com_R = loss_com_R * self.opt.lambda_com_R
#         else:
#             raise ValueError('Unknown commutative loss type.')
#
#         # l1 depth syn
#         if self.opt.l1syndepth_loss == 'usual':
#             loss_l1_DS = self.crtiterionCom_D_DS(self.gen_depth_s[4], self.depth_s.detach())
#             self.loss_l1_DS = loss_l1_DS * self.opt.lambda_l1_DS
#         elif self.opt.l1syndepth_loss == 'pyramid':
#             size = len(self.gen_depth_fake_r)
#             depth_syn = task.scale_pyramid(self.depth_s, size)
#             loss_l1_DS = 0.0
#             for (i_gen_depth_s, i_depth_syn) in zip(self.gen_depth_s, depth_syn):
#                 loss_l1_DS = loss_l1_DS + self.crtiterionCom_D_DS(i_gen_depth_s, i_depth_syn)
#             self.loss_l1_DS = loss_l1_DS * self.opt.lambda_l1_DS
#         else:
#             raise ValueError('Unknown depth l1 loss type.')
#
#         # task loss
#         self.loss_T = self.loss_com_S + self.loss_com_R + self.loss_l1_DS
#         # total loss
#         self.loss_total = self.loss_G * self.opt.lambda_cycle + self.loss_T
#         self.loss_total.backward()
#
#     # def backward_T(self):
#     #     tmp1 = self.gen_depth_s[4].clone()
#     #     self.loss_DS2R_DS = self.crtiterionCom_DS2R_DS(self.gen_depth_fake_r[4], tmp1.detach())
#     #     tmp2 = self.gen_depth_fake_r[4].clone()
#     #     self.loss_DS_DS2R = self.crtiterionCom_DR2S_DR(self.gen_depth_s[4], tmp2.detach())
#     #     self.loss_com_S =  0.5 * (self.loss_DS2R_DS + self.loss_DS_DS2R) * self.opt.lambda_com_S
#     #
#     #     tmp3 = self.gen_depth_r[4].clone()
#     #     self.loss_DR2S_DR = self.crtiterionCom_DS_DS2R(self.gen_depth_fake_s[4], tmp3.detach())
#     #     tmp4 = self.gen_depth_fake_s[4].clone()
#     #     self.loss_DR_DR2S =  self.crtiterionCom_DR_DR2S(self.gen_depth_r[4], tmp4.detach())
#     #     self.loss_com_R = 0.5 * (self.loss_DR2S_DR + self.loss_DR_DR2S) * self.opt.lambda_com_R
#     #
#     #     self.loss_D_DS = self.crtiterionCom_D_DS(self.gen_depth_s[4], self.depth_s.detach())
#     #     self.loss_l1_DS = self.loss_D_DS * self.opt.lambda_l1_DS
#     #
#     #     self.loss_T = self.loss_com_S + self.loss_com_R + self.loss_l1_DS
#     #     self.loss_T.backward()


    def optimize_parameters(self, epoch_iter):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.net_D_S, self.net_D_R], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_T2.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_T2.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.net_D_S, self.net_D_R], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_S()  # calculate gradients for D_A
        self.backward_D_R()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights