from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # training epoch
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count')
        self.parser.add_argument('--niter', type=int, default=6,
                                 help='# of iter with initial learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=4,
                                 help='# of iter to decay learning rate to zero')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--transform_epoch', type=int, default=0,
                                 help='# of epoch for transform learning')
        self.parser.add_argument('--task_epoch', type=int, default=0,
                                 help='# of epoch for task learning')
        # learning rate and loss weight
        self.parser.add_argument('--lr_policy', type=str, default='lambda',
                                 help='learning rate policy[lambda|step|plateau]')
        self.parser.add_argument('--lr_task', type=float, default=1e-4,
                                 help='initial learning rate for adam')
        self.parser.add_argument('--lr_trans', type=float, default=5e-5,
                                 help='initial learning rate for discriminator')
        self.parser.add_argument('--lambda_rec_img', type=float, default=100.0,
                                 help='weight for image reconstruction loss')
        self.parser.add_argument('--lambda_gan_img', type=float, default=1.0,
                                 help='weight for image GAN loss')
        self.parser.add_argument('--lambda_gan_feature', type=float, default=0.1,
                                 help='weight for feature GAN loss')
        self.parser.add_argument('--lambda_rec_lab', type=float, default=100.0,
                                 help='weight for task loss')
        self.parser.add_argument('--lambda_smooth', type=float, default=0.1,
                                 help='weight for depth smooth loss')
        # display the results
        self.parser.add_argument('--display_freq', type=int, default=10,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=10,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results')
        # others
        self.parser.add_argument('--separate', action='store_true',
                                 help='transform and task network training end-to-end or separate')
        self.parser.add_argument('--pool_size', type=int, default=20,
                                 help='the size of image buffer that stores previously generated images')

        self.isTrain = True

        # ADDED
        # CycleGAN options
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        self.parser.add_argument('--lambda_S', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_R', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5,
                            help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        # Commutative options
        self.parser.add_argument('--lambda_com_S', type=float, default=10.0, help='weight for loss D(s) = D(S2R(s))')
        self.parser.add_argument('--lambda_com_R', type=float, default=10.0, help='weight for loss D(r) = D(R2S(r))')
        self.parser.add_argument('--lambda_l1_DS', type=float, default=1.0, help='weight for loss D(s) = D_s')

        self.parser.add_argument('--lambda_cycle', type=float, default=1.0, help='weight for the overall cycle loss in the total loss')
        self.parser.add_argument('--com_loss', type=str, default='usual', help='pyramid|usual')
        self.parser.add_argument('--l1syndepth_loss', type=str, default='usual', help='pyramid|usual')

        # Segmentation options
        self.parser.add_argument('--lr_seg', type=float, default=1e-3,
                                 help='initial learning rate for sgd')
        self.parser.add_argument('--momentum_seg', type=float, default=0.9,
                                 help='momentum for sgd for the segmentation task')
        self.parser.add_argument('--lambda_seg_S', type=float, default=10.0,
                                 help='weight for commutative segmentation loss for realistic images')
        self.parser.add_argument('--lambda_seg_com_S', type=float, default=10.0,
                                 help='weight for commutative segmentation loss for realistic images')
        self.parser.add_argument('--lambda_seg_com_R', type=float, default=10.0,
                                 help='weight for commutative segmentation loss for synthetic images')




