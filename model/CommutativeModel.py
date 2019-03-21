import torch
from torch.autograd import Variable
import util.task as task
from .base_model import BaseModel
from . import network
from .import networks_cg


class CommutativeModel(BaseModel):
    def name(self):
        return 'Commutative Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # self.loss_names = ['lab_s', 'lab_t', 'lab_smooth']
        self.loss_names = ['lab_s', 'lab_t']
        self.visual_names = ['img_s', 'lab_s', 'lab_s_g', 'img_t', 'lab_t', 'lab_t_g']
        self.model_names = ['img2task']

        # define the task network
        self.net_img2task = network.define_G(opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
                                           opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
                                           False, opt.gpu_ids, opt.U_weight)
        #define the S->R and R->S networks
        self.net_syn2real = networks_cg.define_G(opt.image_nc, opt.image_nc, opt.ngf, 'resnet_9blocks', norm='instance',
                                                 gpu_ids=self.gpu_ids)
        self.net_real2syn = networks_cg.define_G(opt.image_nc, opt.image_nc, opt.ngf, 'resnet_9blocks', norm='instance',
                                                 gpu_ids=self.gpu_ids)

        if opt.init_models:
            # load task net
            self.net_img2task.load_state_dict(torch.load(opt.init_depth_netG_filename))

            # load syn2real net
            if isinstance(self.net_syn2real, torch.nn.DataParallel):
                print('self.syn2real')
                self.net_syn2real = self.net_syn2real.module
            state_dict = torch.load(opt.init_syn2real_netG_filename, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            self.net_syn2real.load_state_dict(state_dict)
            # self.net_syn2real.load_state_dict(torch.load(opt.init_syn2real_netG_filename))


            # load real2syn net
            if isinstance(self.net_real2syn, torch.nn.DataParallel):
                print('self.real2syn')
                self.net_real2syn = self.net_real2syn.module
            state_dict = torch.load(opt.init_real2syn_netG_filename, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            self.net_real2syn.load_state_dict(state_dict)

            print('Translators and the task net loaded')

        if self.isTrain:

            self.netD_syn = networks_cg.define_D(opt.image_nc, 64, 'basic', 3, 'instance', 'normal', 0.02, self.gpu_ids)
            self.netD_real = networks_cg.define_D(opt.image_nc, 64, 'basic', 3, 'instance', 'normal', 0.02,
                                                  self.gpu_ids)
            if opt.init_models:
                if isinstance(self.netD_syn, torch.nn.DataParallel):
                    print('netD_syn')
                    self.netD_syn = self.netD_syn.module
                state_dict = torch.load(opt.init_real2syn_netD_filename, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                self.netD_syn.load_state_dict(state_dict)

                if isinstance(self.netD_real, torch.nn.DataParallel):
                    print('netD_real')
                    self.netD_real = self.netD_real.module
                state_dict = torch.load(opt.init_syn2real_netD_filename, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                self.netD_real.load_state_dict(state_dict)

            # define the loss function
            self.l1loss = torch.nn.L1Loss()
            self.l2loss = torch.nn.MSELoss()

            self.optimizer_img2task = torch.optim.Adam(self.net_img2task.parameters(), lr=opt.lr_task,
                                                       betas=(0.9, 0.999))

            self.optimizers = []
            self.schedulers = []

            self.optimizers.append(self.optimizer_img2task)
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
        self.img_t = Variable(self.img_target)
        self.lab_s = Variable(self.lab_source)
        self.lab_t = Variable(self.lab_target)

    def foreward_G_basic(self, net_G, img_s, img_t):

        img = torch.cat([img_s, img_t], 0)
        fake = net_G(img)

        size = len(fake)

        f_s, f_t = fake[0].chunk(2)
        img_fake = fake[1:]

        img_s_fake = []
        img_t_fake = []

        for img_fake_i in img_fake:
            img_s, img_t = img_fake_i.chunk(2)
            img_s_fake.append(img_s)
            img_t_fake.append(img_t)

        return img_s_fake, img_t_fake, f_s, f_t, size

    def backward_task(self):

        self.lab_s_g, self.lab_t_g, self.lab_f_s, self.lab_f_t, size = \
            self.foreward_G_basic(self.net_img2task, self.img_s, self.img_t)

        lab_real = task.scale_pyramid(self.lab_s, size-1)
        task_loss = 0
        for (lab_fake_i, lab_real_i) in zip(self.lab_s_g, lab_real):
            task_loss += self.l1loss(lab_fake_i, lab_real_i)

        self.loss_lab_s = task_loss * self.opt.lambda_rec_lab

        img_real = task.scale_pyramid(self.img_t, size-1)
        self.loss_lab_smooth = task.get_smooth_weight(self.lab_t_g, img_real, size-1) * self.opt.lambda_smooth

        # total_loss = self.loss_lab_s + self.loss_lab_smooth
        total_loss = task_loss

        total_loss.backward()

    def optimize_parameters(self, epoch_iter):

        self.forward()
        # task network
        self.optimizer_img2task.zero_grad()
        self.backward_task()
        self.optimizer_img2task.step()

    def validation_target(self):

        lab_real = task.scale_pyramid(self.lab_t, len(self.lab_t_g))
        task_loss = 0
        for (lab_fake_i, lab_real_i) in zip(self.lab_t_g, lab_real):
            task_loss += self.l1loss(lab_fake_i, lab_real_i)

        self.loss_lab_t = task_loss * self.opt.lambda_rec_lab
