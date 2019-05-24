import torch
import itertools
import util.task as task
from torch.autograd import Variable
from .base_model import BaseModel
from util.image_pool import ImagePool
from . import network
from . import networks_cg


class CommutativeModel(BaseModel):
    def name(self):
        return 'Commutative Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['G_R2S', 'G_S2R', 'D_S', 'D_R', 'cycle_S', 'cycle_R', 'com_S', 'com_R', 'l1_DS']
        self.visual_names_S = ['img_s', 'fake_img_r', 'gen_depth_s', 'depth_s', 'gen_depth_fake_r']
        self.visual_names_R = ['img_r', 'fake_img_s', 'gen_depth_r', 'depth_r', 'gen_depth_fake_s']
        self.visual_names_OTHERS = ['rec_img_r', 'rec_img_s', 'idt_img_r', 'idt_img_s']
        if self.isTrain:
            self.model_names = ['Depth', 'S2R', 'R2S', 'D_S', 'D_R']
        else:
            self.model_names = ['Depth', 'S2R', 'R2S']

        self.visual_names = self.visual_names_S + self.visual_names_R

        # define the task network
        self.net_Depth = network.define_G(opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
                                           opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
                                           False, self.gpu_ids, opt.U_weight)

        # print('PARAMS!!')
        # print(opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
        #                                    opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
        #                                    False, self.gpu_ids, opt.U_weight)
        # print('PARAMS!!')
        #define the S->R and R->S networks
        self.net_S2R = networks_cg.define_G(opt.image_nc, opt.image_nc, opt.ngf, 'resnet_9blocks', norm='instance',
                                                 gpu_ids=self.gpu_ids)
        self.net_R2S = networks_cg.define_G(opt.image_nc, opt.image_nc, opt.ngf, 'resnet_9blocks', norm='instance',
                                                 gpu_ids=self.gpu_ids)

        if opt.init_models:
            # load task net
            self.net_Depth.load_state_dict(torch.load(opt.init_Depth_netG_filename))
            self.net_Depth = torch.nn.DataParallel(self.net_Depth, self.gpu_ids)

            # load S2R net
            if isinstance(self.net_S2R, torch.nn.DataParallel):
                self.net_S2R = self.net_S2R.module
            state_dict = torch.load(opt.init_S2R_netG_filename, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            self.net_S2R.load_state_dict(state_dict)
            self.net_S2R = torch.nn.DataParallel(self.net_S2R, self.gpu_ids)
            # self.net_S2R.load_state_dict(torch.load(opt.init_syn2real_netG_filename))

            # load R2S net
            if isinstance(self.net_R2S, torch.nn.DataParallel):
                self.net_R2S = self.net_R2S.module
            state_dict = torch.load(opt.init_R2S_netG_filename, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            self.net_R2S.load_state_dict(state_dict)
            self.net_R2S = torch.nn.DataParallel(self.net_R2S, self.gpu_ids)

            print('Translators and the task net loaded')

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
                self.net_D_S = torch.nn.DataParallel(self.net_D_S, self.gpu_ids)

                if isinstance(self.net_D_R, torch.nn.DataParallel):
                    self.net_D_R = self.net_D_R.module
                state_dict = torch.load(opt.init_S2R_netD_filename, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                self.net_D_R.load_state_dict(state_dict)
                self.net_D_R = torch.nn.DataParallel(self.net_D_R, self.gpu_ids)

                print('Discriminator nets loaded')

            # create image buffers to store previously generated images
            self.fake_img_s_pool = ImagePool(opt.batchSize)
            self.fake_img_r_pool = ImagePool(opt.batchSize)

            # define loss functions
                # cyclegan losses
            self.criterionGAN = networks_cg.GANLoss(self.opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
                # depth losses
            self.crtiterionCom_DS2R_DS = torch.nn.L1Loss()
            self.crtiterionCom_DR2S_DR = torch.nn.L1Loss()
            self.crtiterionCom_DS_DS2R = torch.nn.L1Loss()
            self.crtiterionCom_DR_DR2S = torch.nn.L1Loss()
            self.crtiterionCom_D_DS = torch.nn.L1Loss()

            # define optimizers
            # self.optimizer_T2 = torch.optim.Adam(itertools.chain(self.net_Depth.parameters(), self.net_S2R.parameters(),
            #                                                      self.net_R2S.parameters()),
            #                                     lr=0.0002, betas=(0.5, 0.9))
            #
            # self.optimizer_D = torch.optim.Adam(
            #     itertools.chain(self.net_D_S.parameters(), self.net_D_R.parameters()),
            #     lr=0.0002, betas=(0.5, 0.9))

            self.optimizer_T2 = torch.optim.Adam(
                [{'params': self.net_Depth.parameters(), 'lr': opt.lr_task, 'betas': (0.95, 0.999)},
                 {'params': itertools.chain(self.net_S2R.parameters(), self.net_R2S.parameters())}
                ],
                lr=opt.lr_trans, betas=(0.5, 0.9))

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

    def set_input(self, input):
        self.input = input
        self.img_source = input['img_source']
        self.img_target = input['img_target']
        if self.isTrain:
            self.lab_source = input['lab_source']
            self.lab_target = input['lab_target']

        if len(self.gpu_ids) > 0:
            self.img_source = self.img_source.cuda(self.gpu_ids[0], async=True)
            self.img_target = self.img_target.cuda(self.gpu_ids[0], async=True)
            if self.isTrain:
                self.lab_source = self.lab_source.cuda(self.gpu_ids[0], async=True)
                self.lab_target = self.lab_target.cuda(self.gpu_ids[0], async=True)

    def forward(self):
        self.img_s = Variable(self.img_source)
        self.img_r = Variable(self.img_target)
        self.depth_s = Variable(self.lab_source)
        self.depth_r = Variable(self.lab_target)

        self.fake_img_s = self.net_R2S(self.img_r)
        self.fake_img_r = self.net_S2R(self.img_s)
        self.rec_img_s = self.net_R2S(self.fake_img_r)
        self.rec_img_r = self.net_S2R(self.fake_img_s)

        # identity term
        self.idt_img_s = self.net_R2S(self.img_s)
        self.idt_img_r = self.net_S2R(self.img_r)

        self.gen_depth_s = self.net_Depth(self.img_s)
        self.gen_depth_r = self.net_Depth(self.img_r)
        self.gen_depth_fake_s = self.net_Depth(self.fake_img_s)
        self.gen_depth_fake_r = self.net_Depth(self.fake_img_r)

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
        fake_s = self.fake_img_s_pool.query(self.fake_img_s)
        self.loss_D_S = self.backward_D_basic(self.net_D_S, self.img_s, fake_s)

    def backward_D_R(self):
        fake_r = self.fake_img_r_pool.query(self.fake_img_r)
        self.loss_D_R = self.backward_D_basic(self.net_D_R, self.img_r, fake_r)

    def backward_G_cycle(self):
        # CycleGAN loss
        lambda_idt = self.opt.lambda_identity
        lambda_S = self.opt.lambda_S    # syn
        lambda_R = self.opt.lambda_R    # real
        # GAN loss D_A(G_A(A))
        self.loss_G_R2S = self.criterionGAN(self.net_D_S(self.fake_img_s), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_S2R = self.criterionGAN(self.net_D_R(self.fake_img_r), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_S = self.criterionCycle(self.rec_img_s, self.img_s) * lambda_S
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_R = self.criterionCycle(self.rec_img_r, self.img_r) * lambda_R

        # Identity loss
        if lambda_idt > 0:
            self.loss_idt_S = self.criterionIdt(self.idt_img_s, self.img_s) * lambda_S * lambda_idt
            self.loss_idt_R = self.criterionIdt(self.idt_img_r, self.img_r) * lambda_R * lambda_idt
        else:
            self.loss_idt_S = 0
            self.loss_idt_R = 0

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_R2S + self.loss_G_S2R + self.loss_cycle_S \
                      + self.loss_cycle_R + self.loss_idt_S + self.loss_idt_R
        self.loss_G = self.loss_G * self.opt.lambda_cycle
        self.loss_G.backward(retain_graph=True)


    def backward_G_task(self):
        if self.opt.com_loss == 'usual':
            # com S
            loss_com_S = torch.mean(torch.abs(self.gen_depth_fake_r[4] - self.gen_depth_s[4]))
            self.loss_com_S = loss_com_S * self.opt.lambda_com_S
            # com R
            loss_com_R = torch.mean(torch.abs(self.gen_depth_fake_s[4] - self.gen_depth_r[4]))
            self.loss_com_R = loss_com_R * self.opt.lambda_com_R
        elif self.opt.com_loss == 'pyramid':
            # com S
            loss_com_S = 0.0
            for (i_gen_depth_fake_r, i_gen_depth_s) in zip(self.gen_depth_fake_r, self.gen_depth_s):
                loss_com_S = loss_com_S + torch.mean(torch.abs(i_gen_depth_fake_r - i_gen_depth_s))
            self.loss_com_S = loss_com_S * self.opt.lambda_com_S
            # com R
            loss_com_R = 0.0
            for (i_gen_depth_fake_s, i_gen_depth_r) in zip(self.gen_depth_fake_s, self.gen_depth_r):
                loss_com_R = loss_com_R + torch.mean(torch.abs(i_gen_depth_fake_s - i_gen_depth_r))
            self.loss_com_R = loss_com_R * self.opt.lambda_com_R
        else:
            raise ValueError('Unknown commutative loss type.')

        # l1 depth syn
        if self.opt.l1syndepth_loss == 'usual':
            loss_l1_DS = self.crtiterionCom_D_DS(self.gen_depth_s[4], self.depth_s.detach())
            self.loss_l1_DS = loss_l1_DS * self.opt.lambda_l1_DS
        elif self.opt.l1syndepth_loss == 'pyramid':
            size = len(self.gen_depth_fake_r)
            depth_syn = task.scale_pyramid(self.depth_s, size)
            loss_l1_DS = 0.0
            for (i_gen_depth_s, i_depth_syn) in zip(self.gen_depth_s, depth_syn):
                loss_l1_DS = loss_l1_DS + self.crtiterionCom_D_DS(i_gen_depth_s, i_depth_syn)
            self.loss_l1_DS = loss_l1_DS * self.opt.lambda_l1_DS
        else:
            raise ValueError('Unknown depth l1 loss type.')

        # task loss
        self.loss_T = self.loss_com_S + self.loss_com_R + self.loss_l1_DS
        self.loss_T.backward()


    def optimize_parameters(self, epoch_iter):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.net_D_S, self.net_D_R], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_T2.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G_cycle()  # calculate gradients for G_A and G_B
        self.backward_G_task()  # calculate gradients for G_T
        self.optimizer_T2.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.net_D_S, self.net_D_R], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_S()  # calculate gradients for D_A
        self.backward_D_R()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights