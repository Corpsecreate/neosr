import os
import time
import random
from collections import OrderedDict
from copy import deepcopy
from os import path as osp

import numpy as np
import torch
import pytorch_optimizer
from tqdm import tqdm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.nn import functional as F

from neosr.archs import build_network
from neosr.losses import build_loss
from neosr.optimizers import build_optimizer
from neosr.losses.wavelet_guided import wavelet_guided
from neosr.losses.loss_util import get_refined_artifact_map
from neosr.data.augmentations import apply_augment
from neosr.metrics import calculate_metric

from neosr.utils import get_root_logger, imwrite, tensor2img
from neosr.utils.dist_util import master_only
from neosr.utils.registry import MODEL_REGISTRY

from torch.autograd import Variable
from torch import autograd

@MODEL_REGISTRY.register()
class default():
    """Default model."""

    def __init__(self, opt):
    
        self.opt        = opt
        self.device     = torch.device('cuda')
        self.is_train   = opt['is_train']
        self.optimizers = []
        self.schedulers = []

        # define network net_g
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        if self.opt.get('print_network', False) is True:
            self.print_network(self.net_g)

        # define network net_d
        self.net_d = self.opt.get('network_d', None)
        if self.net_d is not None:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            if self.opt.get('print_network', False) is True:
                self.print_network(self.net_d)

        # load pretrained g
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g')
            self.load_network(self.net_g, load_path, param_key, self.opt['path'].get(
                'strict_load_g', True))

        # load pretrained d
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d')
            self.load_network(self.net_d, load_path, param_key, self.opt['path'].get(
                'strict_load_d', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):

        # options var
        train_opt = self.opt['train']
        self.normalise_losses = self.opt.get("normalise_losses", False)
        self.loss_alpha = 0.9985
        self.loss_emas  = {}
        self.prev_emas  = {}
        self.live_emas  = {}
        self.grad_vecs  = {}
        self.log_dict   = {}
        
        self.optimise_calls = 0
        self.optimise_perf = 0
        self.optimise_time = 0

        # set nets to training mode
        self.net_g.train()
        if self.opt.get('network_d', None) is not None:
            self.net_d.train()
        
        self.net_d_iters      = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.optim_to_sched = {s.optimizer : s for s in self.schedulers}
        self.sched_to_optim = {s : s.optimizer for s in self.schedulers}

        self.grad_updates = 0
        self.grad_updates_g = 0
        self.grad_updates_d = 0
        self.loss_samples = 0
        self.scaler_g     = torch.cuda.amp.GradScaler(enabled=True, init_scale=2.**20, growth_interval=1000)
        self.scaler_d     = torch.cuda.amp.GradScaler(enabled=True, init_scale=2.**20, growth_interval=1000)
           
        # initialise counter of how many batches has to be accumulated
        self.accum_count   = 0
        self.accum_count_g = 0
        self.accum_count_d = 0
        self.accum_limit   = self.opt["datasets"]["train"].get("accumulate", 1) 

        # scale ratio var
        self.scale = self.opt['scale'] 

        # gt size var
        if self.opt["model_type"] == "otf":
            self.gt_size = self.opt["gt_size"]
        else:
            self.gt_size = self.opt["datasets"]["train"].get("gt_size")

        # augmentations
        self.aug      = self.opt["datasets"]["train"].get("augmentation", None)
        self.aug_prob = self.opt["datasets"]["train"].get("aug_prob", None)
            
        # for amp
        self.use_amp   = self.opt.get('use_amp', False) is True
        self.amp_dtype = torch.bfloat16 if self.opt.get('bfloat16', False) is True else torch.float16

        # LQ matching for Color/Luma losses
        self.match_lq = self.opt['train'].get('match_lq', False)
        
        self.all_losses = []

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_pix = None
        self.all_losses.append(self.cri_pix)

        if train_opt.get('mssim_opt'):
            self.cri_mssim = build_loss(train_opt['mssim_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_mssim = None
        self.all_losses.append(self.cri_mssim)

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_perceptual = None
        self.all_losses.append(self.cri_perceptual)
        
        if train_opt.get('dists_opt'):
            self.cri_dists = build_loss(train_opt['dists_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_dists = None
        self.all_losses.append(self.cri_dists)
        
        # GAN loss
        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_gan = None
        self.all_losses.append(self.cri_gan)
        
        # LDL loss
        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_ldl = None
        self.all_losses.append(self.cri_ldl)
        
        # Focal Frequency Loss
        if train_opt.get('ff_opt'):
            self.cri_ff = build_loss(train_opt['ff_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_ff = None
        self.all_losses.append(self.cri_ff)
        
        # Gradient-Weighted loss
        if train_opt.get('gw_opt'):
            self.cri_gw = build_loss(train_opt['gw_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_gw = None
        self.all_losses.append(self.cri_gw)
        
        # Color loss
        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_color = None
        self.all_losses.append(self.cri_color)
        
        # Luma loss
        if train_opt.get('luma_opt'):
            self.cri_luma = build_loss(train_opt['luma_opt']).to(self.device, memory_format=torch.channels_last, non_blocking=True)
        else:
            self.cri_luma = None
        self.all_losses.append(self.cri_luma)

        # Wavelet Guided loss
        self.wavelet_guided = self.opt["train"].get("wavelet_guided", "off")
        if self.wavelet_guided == "on" or self.wavelet_guided == "disc":
            logger = get_root_logger()
            logger.info('Loss [Wavelet-Guided] enabled.')
            self.wg_pw    = train_opt.get("wg_pw", 0.01)
            self.wg_pw_lh = train_opt.get("wg_pw_lh", 0.01)
            self.wg_pw_hl = train_opt.get("wg_pw_hl", 0.01)
            self.wg_pw_hh = train_opt.get("wg_pw_hh", 0.05)
            
        self.num_g_losses = sum([1 for loss in self.all_losses if (loss is not None and loss.loss_weight != 0)])

        # gradient clipping
        self.gradclip = self.opt["train"].get("grad_clip", True)

        # error handling
        optim_d            = self.opt["train"].get("optim_d", None)
        pix_losses_bool    = self.cri_pix or self.cri_mssim is not None
        percep_losses_bool = self.cri_perceptual or self.cri_dists is not None

        if pix_losses_bool is False and percep_losses_bool is False:
            raise ValueError('Both pixel/mssim and perceptual losses are None. Please enable at least one.')
        if self.wavelet_guided == "on":
            if self.cri_perceptual is None and self.cri_dists is None:
                msg = "Please enable at least one perceptual loss with weight =>1.0 to use Wavelet Guided"
                raise ValueError(msg)
        if self.net_d is None and optim_d is not None:
            msg = "Please set a discriminator in network_d or disable optim_d."
            raise ValueError(msg)
        if self.net_d is not None and optim_d is None:
            msg = "Please set an optimizer for the discriminator or disable network_d."
            raise ValueError(msg)
        if self.net_d is not None and self.cri_gan is None:
            msg = "Discriminator needs GAN to be enabled."
            raise ValueError(msg)
        if self.net_d is None and self.cri_gan is not None:
            msg = "GAN requires a discriminator to be set."
            raise ValueError(msg)
        if self.aug is not None and self.gt_size % 4 != 0:
            msg = "The gt_size value must be a multiple of 4. Please change it."
            raise ValueError(msg)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        # uppercase optim_type to make it case insensitive
        optim_type_upper = optim_type.upper()
        optim_map = {"ADADELTA"   : torch.optim.Adadelta,
                     "ADAGRAD"    : torch.optim.Adagrad,
                     "ADAM"       : torch.optim.Adam,
                     "ADAMW"      : torch.optim.AdamW,
                     "SPARSEADAM" : torch.optim.SparseAdam,
                     "ADAMAX"     : torch.optim.Adamax,
                     "ASGD"       : torch.optim.ASGD,
                     "SGD"        : torch.optim.SGD,
                     "RADAM"      : torch.optim.RAdam,
                     "RPROP"      : torch.optim.Rprop,
                     "RMSPROP"    : torch.optim.RMSprop,
                     "NADAM"      : torch.optim.NAdam,
                     "LBFGS"      : torch.optim.LBFGS,
                     "ADAN"       : pytorch_optimizer.Adan,
                     "LAMB"       : pytorch_optimizer.Lamb,
                     "LION"       : pytorch_optimizer.Lion,
                    }
        if optim_type_upper in optim_map:
            optimizer = optim_map[optim_type_upper](params, lr, **kwargs)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        if self.opt.get('network_d', None) is not None:
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(
                optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        # uppercase scheduler_type to make it case insensitive
        sch_typ_upper = scheduler_type.upper()
        sch_map = {"CONSTANTLR"        : torch.optim.lr_scheduler.ConstantLR,
                   "LINEARLR"          : torch.optim.lr_scheduler.LinearLR,
                   "EXPONENTIALLR"     : torch.optim.lr_scheduler.ExponentialLR,
                   "CYCLICLR"          : torch.optim.lr_scheduler.CyclicLR,
                   "STEPLR"            : torch.optim.lr_scheduler.StepLR,
                   "MULTISTEPLR"       : torch.optim.lr_scheduler.MultiStepLR,
                   "LAMBDALR"          : torch.optim.lr_scheduler.LambdaLR,
                   "MULTIPLICATIVELR"  : torch.optim.lr_scheduler.MultiplicativeLR,
                   "SEQUENTIALLR"      : torch.optim.lr_scheduler.SequentialLR,
                   "CHAINEDSCHEDULER"  : torch.optim.lr_scheduler.ChainedScheduler,
                   "ONECYCLELR"        : torch.optim.lr_scheduler.OneCycleLR,
                   "POLYNOMIALLR"      : torch.optim.lr_scheduler.PolynomialLR,
                   "CAWR"              : torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                   "COSINEANNEALING"   : torch.optim.lr_scheduler.CosineAnnealingLR,
                   "REDUCELRONPLATEAU" : torch.optim.lr_scheduler.ReduceLROnPlateau,
                  }
        if sch_typ_upper in sch_map:
            for optimizer in self.optimizers:
                self.schedulers.append(sch_map[sch_typ_upper](optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warm-up.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l, strict=True):
            for param_group, lr in zip(optimizer.param_groups, lr_groups, strict=True):
                param_group['lr'] = lr
                
    def get_grad_vector(self, model):
    
        # Calculate the total number of elements for all trainable gradients
        total_params = sum(p.numel() for p in model.parameters() if p.grad is not None and p.requires_grad)

        # Allocate a single tensor to hold all gradients
        gradients_1d = torch.zeros(total_params)

        # Directly fill the allocated tensor
        offset = 0
        with torch.inference_mode():
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None and param.requires_grad:
                        numel = param.grad.numel()
                        gradients_1d[offset:offset + numel].copy_(param.grad.reshape(-1))
                        offset += numel

        return gradients_1d

    def optimize_parameters(self, current_iter, tb_logger = None):

        self.optimise_calls += 1
        optimise_start_perf = time.perf_counter()
        optimise_start_time = time.process_time()
        turn_g = True
        turn_d = True
        #turn_g = current_iter % 2 == 0
        #turn_d = not turn_g
        
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False
            
        n_samples = self.gt.shape[0]
        self.loss_samples += n_samples
        # increment accumulation counter and check if accumulation limit has been reached
        self.accum_count += 1
        self.accum_count_g += turn_g
        self.accum_count_d += turn_d
        
        # list of loss functions to apply backward() on
        back_losses_g = {}
        trim_size     = 0

        def trim_image(x, trim):
            if trim == 0:
                return x
            return x[:, :, trim : x.shape[2] - trim, trim : x.shape[3] - trim]
            
        #lq_noise = 0.0
        #gt_noise = 0.0
        lq_noise  = torch.normal(0.0, 0.05, self.lq.shape).cuda()
        gt_noise  = torch.normal(0.0, 0.05, self.gt.shape).cuda()
        modded_lq = torch.clamp(self.lq + lq_noise, 0.0, 1.0).detach()
        modded_gt = torch.clamp(self.gt + gt_noise, 0.0, 1.0).detach()
        
        ##########################
        # GENERATOR
        ##########################
        with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):

            self.output   = self.net_g(modded_lq)
            trim_required = (self.gt.shape[-1] - self.output.shape[-1]) // 2
            trimmed_gt = trim_image(modded_gt, trim_required)
            #modded_gt = trim_image(modded_gt, trim_required)
            
            #lq_noise = torch.normal(0.0, 0.05, self.output.shape).cuda()
            #gt_noise = torch.normal(0.0, 0.05, trimmed_gt.shape).cuda()
            #modded_gt = torch.clamp(trimmed_gt + gt_noise, 0.0, 1.0)
            
            if self.grad_updates == 0:
                print("Trimed GT Size: {}".format(str(trimmed_gt.shape)))
                with torch.inference_mode():
                    print(trimmed_gt.min(), trimmed_gt.max())
                
            # lq match
            lq_interp = None

            # wavelet guided loss
            if self.wavelet_guided.upper() in ("ON", "DISC"):
                (
                    LL,
                    LH,
                    HL,
                    HH,
                    combined_HF,
                    LL_gt,
                    LH_gt,
                    HL_gt,
                    HH_gt,
                    combined_HF_gt,
                ) = wavelet_guided(self.output, trimmed_gt)

            l_g_total = 0.0
            loss_dict = OrderedDict()
            curr_vec  = None if self.accum_count_g == 1 else self.get_grad_vector(self.net_g)

            losses_applied = 0
            if (turn_g and current_iter > self.net_d_init_iters and current_iter % self.net_d_iters == 0):
            
                # pixel loss
                if self.cri_pix and self.cri_pix.loss_weight != 0:
                    if self.wavelet_guided == "on":
                        l_g_pix    = self.wg_pw    * self.cri_pix(LL, LL_gt)
                        l_g_pix_lh = self.wg_pw_lh * self.cri_pix(LH, LH_gt)
                        l_g_pix_hl = self.wg_pw_hl * self.cri_pix(HL, HL_gt)
                        l_g_pix_hh = self.wg_pw_hh * self.cri_pix(HH, HH_gt)
                        loss_pix   = (l_g_pix + l_g_pix_lh + l_g_pix_hl + l_g_pix_hh)
                    else:
                        loss_pix = self.cri_pix(self.output, trimmed_gt)
                    
                    detached_loss             = loss_pix.detach().item()
                    loss_dict['l_g_pix']      = detached_loss
                    self.loss_emas['l_g_pix'] = (self.loss_alpha * self.loss_emas.get('l_g_pix', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                        = 1.0 / abs(self.live_emas.get('l_g_pix', self.loss_emas['l_g_pix']))
                    this_loss                 = (loss_pix * self.cri_pix.loss_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total                += (detached_loss * self.cri_pix.loss_weight) * sf
                    losses_applied           += 1
                    del loss_pix, this_loss
                    new_vec    = self.get_grad_vector(self.net_g)
                    grad_delta = (new_vec if curr_vec is None else (new_vec - curr_vec))
                    self.grad_vecs['l_g_pix'] = grad_delta + self.grad_vecs.get('l_g_pix', 0.0)
                    curr_vec = new_vec
                    
                # ssim loss
                if self.cri_mssim and self.cri_mssim.loss_weight != 0:
                    if self.wavelet_guided == "on":
                        l_g_mssim    = self.wg_pw    * self.cri_mssim(LL, LL_gt)
                        l_g_mssim_lh = self.wg_pw_lh * self.cri_mssim(LH, LH_gt)
                        l_g_mssim_hl = self.wg_pw_hl * self.cri_mssim(HL, HL_gt)
                        l_g_mssim_hh = self.wg_pw_hh * self.cri_mssim(HH, HH_gt)
                        loss_mssim   = (l_g_mssim + l_g_mssim_lh + l_g_mssim_hl + l_g_mssim_hh)
                    else:
                        loss_mssim = self.cri_mssim(self.output, trimmed_gt)
                       
                    detached_loss               = loss_mssim.detach().item()
                    loss_dict['l_g_mssim']      = detached_loss
                    self.loss_emas['l_g_mssim'] = (self.loss_alpha * self.loss_emas.get('l_g_mssim', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                          = 1.0 / abs(self.live_emas.get('l_g_mssim', self.loss_emas['l_g_mssim']))
                    this_loss                   = (loss_mssim * self.cri_mssim.loss_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total                  += (detached_loss * self.cri_mssim.loss_weight) * (sf if self.normalise_losses else 1.0)
                    losses_applied             += 1
                    del loss_mssim, this_loss
                    new_vec    = self.get_grad_vector(self.net_g)
                    grad_delta = (new_vec if curr_vec is None else (new_vec - curr_vec))
                    self.grad_vecs['l_g_mssim'] = grad_delta + self.grad_vecs.get('l_g_mssim', 0.0)
                    curr_vec = new_vec
                    
                # perceptual loss
                if self.cri_perceptual and self.cri_perceptual.perceptual_weight != 0:
                    l_g_percep                  = self.cri_perceptual(self.output, trimmed_gt)
                    
                    detached_loss               = l_g_percep.detach().item()
                    loss_dict['l_g_percep']      = detached_loss
                    self.loss_emas['l_g_percep'] = (self.loss_alpha * self.loss_emas.get('l_g_percep', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                          = 1.0 / abs(self.live_emas.get('l_g_percep', self.loss_emas['l_g_percep']))
                    this_loss                   = (l_g_percep * self.cri_perceptual.perceptual_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total                  += (detached_loss * self.cri_perceptual.perceptual_weight) * (sf if self.normalise_losses else 1.0)
                    losses_applied             += 1
                    del l_g_percep, this_loss
                    new_vec    = self.get_grad_vector(self.net_g)
                    grad_delta = (new_vec if curr_vec is None else (new_vec - curr_vec))
                    self.grad_vecs['l_g_percep'] = grad_delta + self.grad_vecs.get('l_g_percep', 0.0)
                    curr_vec = new_vec
                    
                # dists loss
                if self.cri_dists and self.cri_dists.loss_weight != 0:
                    l_g_dists                   = self.cri_dists(self.output, trimmed_gt)
                    
                    detached_loss               = l_g_dists.detach().item()
                    loss_dict['l_g_dists']      = detached_loss
                    self.loss_emas['l_g_dists'] = (self.loss_alpha * self.loss_emas.get('l_g_dists', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                          = 1.0 / abs(self.live_emas.get('l_g_dists', self.loss_emas['l_g_dists']))
                    this_loss                   = (l_g_dists * self.cri_dists.loss_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total                  += (detached_loss * self.cri_dists.loss_weight) * (sf if self.normalise_losses else 1.0)
                    losses_applied             += 1
                    del l_g_dists, this_loss
                    new_vec    = self.get_grad_vector(self.net_g)
                    grad_delta = (new_vec if curr_vec is None else (new_vec - curr_vec))
                    self.grad_vecs['l_g_dists'] = grad_delta + self.grad_vecs.get('l_g_dists', 0.0)
                    curr_vec = new_vec
                    
                # ldl loss
                if self.cri_ldl and self.cri_ldl.loss_weight != 0:
                    pixel_weight             = get_refined_artifact_map(trimmed_gt, self.output, 7)
                    l_g_ldl                  = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, trimmed_gt))
                    
                    detached_loss             = l_g_ldl.detach().item()
                    loss_dict['l_g_ldl']      = detached_loss
                    self.loss_emas['l_g_ldl'] = (self.loss_alpha * self.loss_emas.get('l_g_ldl', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                        = 1.0 / abs(self.live_emas.get('l_g_ldl', self.loss_emas['l_g_ldl']))
                    this_loss                 = (l_g_ldl * self.cri_ldl.loss_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total                += (detached_loss * self.cri_ldl.loss_weight) * (sf if self.normalise_losses else 1.0)
                    losses_applied           += 1
                    del l_g_ldl, this_loss
                    
                # focal frequency loss
                if self.cri_ff and self.cri_ff.loss_weight != 0:
                    l_g_ff                  = self.cri_ff(self.output, trimmed_gt)
                    
                    detached_loss            = l_g_ff.detach().item()
                    loss_dict['l_g_ff ']     = detached_loss
                    self.loss_emas['l_g_ff'] = (self.loss_alpha * self.loss_emas.get('l_g_ff', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                       = 1.0 / abs(self.live_emas.get('l_g_ff', self.loss_emas['l_g_ff']))
                    this_loss                = (l_g_ff * self.cri_ff.loss_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total               += (detached_loss * self.cri_ff.loss_weight) * (sf if self.normalise_losses else 1.0)
                    losses_applied          += 1
                    del l_g_ff, this_loss
                    
                # gradient-weighted loss
                if self.cri_gw and self.cri_gw.loss_weight != 0:
                    l_g_gw                  = self.cri_gw(self.output, trimmed_gt)
                    
                    detached_loss            = l_g_gw.detach().item()
                    loss_dict['l_g_gw']      = detached_loss
                    self.loss_emas['l_g_gw'] = (self.loss_alpha * self.loss_emas.get('l_g_gw', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                       = 1.0 / abs(self.live_emas.get('l_g_gw', self.loss_emas['l_g_gw']))
                    this_loss                = (l_g_gw * self.cri_gw.loss_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total               += (detached_loss * self.cri_gw.loss_weight) * (sf if self.normalise_losses else 1.0)
                    losses_applied          += 1
                    del l_g_gw, this_loss
                    
                # color loss
                if self.cri_color and self.cri_color.loss_weight != 0:
                    if self.match_lq and lq_interp is None:
                        lq_interp = F.interpolate(self.lq, scale_factor=self.scale, mode='bicubic')
                        lq_interp = trim_image(lq_interp, trim_required)
                    l_g_color                  = self.cri_color(self.output, trimmed_gt if not self.match_lq else lq_interp)
                    
                    detached_loss            = l_g_color.detach().item()
                    loss_dict['l_g_color']      = detached_loss
                    self.loss_emas['l_g_color'] = (self.loss_alpha * self.loss_emas.get('l_g_color', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                       = 1.0 / abs(self.live_emas.get('l_g_color', self.loss_emas['l_g_color']))
                    this_loss                = (l_g_color * self.cri_color.loss_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total               += (detached_loss * self.cri_color.loss_weight) * (sf if self.normalise_losses else 1.0)
                    losses_applied          += 1
                    del l_g_color, this_loss
                    
                # luma loss
                if self.cri_luma and self.cri_luma.loss_weight != 0:
                    if self.match_lq and lq_interp is None:
                        lq_interp = F.interpolate(self.lq, scale_factor=self.scale, mode='bicubic')
                        lq_interp = trim_image(lq_interp, trim_required)
                    
                    l_g_luma                 = self.cri_luma(self.output, trimmed_gt if not self.match_lq else lq_interp)
                    detached_loss            = l_g_luma.detach().item()
                    loss_dict['l_g_luma']      = detached_loss
                    self.loss_emas['l_g_luma'] = (self.loss_alpha * self.loss_emas.get('l_g_luma', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                       = 1.0 / abs(self.live_emas.get('l_g_luma', self.loss_emas['l_g_luma']))
                    this_loss                = (l_g_luma * self.cri_luma.loss_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total               += (detached_loss * self.cri_luma.loss_weight) * (sf if self.normalise_losses else 1.0)
                    losses_applied          += 1
                    del l_g_luma, this_loss
                    
                # GAN loss
                if self.cri_gan and self.cri_gan.loss_weight != 0:
                    
                    if self.cri_gan.gan_type in ("rsgan", "rasgan"):
                        real_pred = self.net_d(modded_gt).detach()
                        fake_pred = self.net_d(self.output)
                        l_g_gan   = self.cri_gan(fake_pred - real_pred, True, is_disc=False)
                    else:
                        fake_g_pred              = self.net_d(self.output)
                        l_g_gan                  = self.cri_gan(fake_g_pred, True, is_disc=False)
                        
                    detached_loss             = l_g_gan.detach().item()
                    loss_dict['l_g_gan']      = detached_loss
                    self.loss_emas['l_g_gan'] = (self.loss_alpha * self.loss_emas.get('l_g_gan', detached_loss)) + (1 - self.loss_alpha) * detached_loss
                    sf                        = 1.0 / abs(self.live_emas.get('l_g_gan', self.loss_emas['l_g_gan']))
                    this_loss                 = (l_g_gan * self.cri_gan.loss_weight) * (sf if self.normalise_losses else 1.0)
                    
                    self.scaler_g.scale(this_loss / self.accum_limit).backward(retain_graph = losses_applied <= self.num_g_losses - 1)
                    l_g_total                += (detached_loss * self.cri_gan.loss_weight) * (sf if self.normalise_losses else 1.0)
                    losses_applied           += 1
                    del l_g_gan, this_loss
                    new_vec    = self.get_grad_vector(self.net_g)
                    grad_delta = (new_vec if curr_vec is None else (new_vec - curr_vec))
                    self.grad_vecs['l_g_gan'] = grad_delta + self.grad_vecs.get('l_g_gan', 0.0)
                    curr_vec = new_vec
        
        #for i, (_, loss) in enumerate(back_losses_g.items()):
        #    is_last_loss = i == len(back_losses_g) - 1
        #    self.scaler_g.scale(loss / self.accum_limit).backward(retain_graph = not is_last_loss)
            
        #if len(back_losses_g) <= 1:
        #    self.normalise_losses = False
            
        # add total generator loss for tensorboard tracking
        loss_dict['l_g_total'] = l_g_total

        ##########################
        # DISCRIMINATOR
        ##########################
        # optimize net_d
        has_net_d = self.net_d is not None and self.cri_gan and self.cri_gan.loss_weight != 0
        if has_net_d and turn_d:
        
            for p in self.net_d.parameters():
                p.requires_grad = True
                
            # add total discriminator loss for tensorboard tracking
            loss_dict['l_d_total'] = 0.0

            with torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            
                #modded_gt = torch.clamp(trimmed_gt + gt_noise, 0.0, 1.0)
                #modded_gt = trimmed_gt
                if self.cri_gan.gan_type in ('rsgan', 'rasgan'):
                    real_d_pred = self.net_d(modded_gt)
                    fake_d_pred = self.net_d(self.output.detach())
                    if self.cri_gan.gan_type == 'rsgan':
                        l_d_real = self.cri_gan(real_d_pred - fake_d_pred, True, is_disc=True)
                        l_d_fake = self.cri_gan(fake_d_pred - real_d_pred, False, is_disc=True)
                    elif self.cri_gan.gan_type == 'rasgan':
                        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True)
                        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred), False, is_disc=True)
                else:
                    # real
                    if self.wavelet_guided == "on" or self.wavelet_guided == "disc":
                        real_d_pred = self.net_d(combined_HF_gt)
                    else:
                        real_d_pred = self.net_d(modded_gt)    
                    l_d_real                = self.cri_gan(real_d_pred, True, is_disc=True)
                    
                    # fake
                    if self.wavelet_guided == "on" or self.wavelet_guided == "disc":
                        fake_d_pred = self.net_d(combined_HF.detach().clone())
                    else:
                        fake_d_pred = self.net_d(self.output.detach().clone())
                    l_d_fake                = self.cri_gan(fake_d_pred, False, is_disc=True)
                    
                l_d_real_item           = l_d_real.detach().item()
                loss_dict['l_d_real']   = l_d_real_item
                loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                loss_dict['l_d_total'] += 0.5 * l_d_real_item
                #self.scaler_d.scale(l_d_real / self.accum_limit).backward()
                #del l_d_real

                l_d_fake_item           = l_d_fake.detach().item()
                loss_dict['l_d_fake']   = l_d_fake_item
                loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                loss_dict['l_d_total'] += 0.5 * l_d_fake_item
                #self.scaler_d.scale(l_d_fake / self.accum_limit).backward()
                #del l_d_fake
                
                self.scaler_d.scale(0.5*(l_d_real + l_d_fake) / self.accum_limit).backward()
                
            #one = torch.FloatTensor([1]).to(self.device, non_blocking=True)
            #mone = one * -1
            # Compute gradients

            #self.scaler_d.scale(l_d_real / self.accum_limit).backward(retain_graph=True)
            #self.scaler_d.scale(-l_d_fake / self.accum_limit).backward(retain_graph=False)
            #self.scaler_d.scale(l_d_fake / self.accum_limit).backward(retain_graph=True)
            #self.scaler_d.scale(l_d_real / self.accum_limit).backward(retain_graph=False)
            
            if self.cri_gan.gan_type == 'wgaasdn':
                batch_size = trimmed_gt.shape[0]
                eta = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(0,1)
                eta = eta.expand(batch_size, trimmed_gt.size(1), trimmed_gt.size(2), trimmed_gt.size(3))
                eta = eta.cuda()

                interpolated = eta * trimmed_gt + ((1 - eta) * self.output)
                interpolated = interpolated.cuda()

                # define it to calculate gradient
                interpolated = Variable(interpolated, requires_grad=True)

                # calculate probability of interpolated examples
                prob_interpolated = self.net_d(interpolated)

                # calculate gradients of probabilities with respect to examples
                gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                       grad_outputs=torch.ones(prob_interpolated.size()).cuda(), create_graph=True, retain_graph=True)[0]

                # flatten the gradients to it calculates norm batchwise
                gradients = gradients.view(gradients.size(0), -1)
                
                grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
                self.scaler_d.scale(grad_penalty).backward()
            
        if self.accum_count >= self.accum_limit:
            self.accum_count   = 0
            self.grad_updates += 1
            
        if self.accum_count_g >= self.accum_limit:
            self.accum_count_g   = 0
            self.grad_updates_g += 1
            
            scale_factor_g = self.scaler_g.get_scale()
            for key in self.grad_vecs:
                grad_1d   = self.grad_vecs[key] / scale_factor_g
                grad_size = torch.sqrt(torch.sum(grad_1d.square()))
                tb_logger.add_scalar(f'gradients/{key}', grad_size, current_iter / self.accum_limit)
                
            grad_1d   = self.get_grad_vector(self.net_g) / scale_factor_g
            grad_size = torch.sqrt(torch.sum(grad_1d.square()))
            tb_logger.add_scalar(f'gradients/l_g_total', grad_size, current_iter / self.accum_limit)
                
            # Generator
            # gradient clipping on generator
            if self.gradclip:
                self.scaler_g.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01, error_if_nonfinite=False)
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
            self.optimizer_g.zero_grad(set_to_none=True)
            self.optim_to_sched[self.optimizer_g].step()
            
            if len(self.prev_emas) == 0:
                self.prev_emas = {k : v for k, v in self.loss_emas.items()}
            
            if self.grad_updates_g > 0 and self.grad_updates_g % 100 == 0:
                print("="*80)
                print("{:<16}{:<13}{:<1}{:<13}{:<13}{:<13}".format("Metric", "Previous", "", "Current", "Change", "Change%"))
                COL_RESET = '\033[0m'
                for metric, loss_val in self.loss_emas.items():
                    prev = self.prev_emas.get(metric, loss_val)
                    chng = loss_val - prev
                    pct  = loss_val / (1 if prev == 0 else prev) - 1
                    sign = '+' if chng > 0 else '-'
                    colcode = '\33[42m' if chng <= 0 else '\33[41m'
                    print("{:<16}{:<13.5f} {:<13.5f}{:<18}{}{:<11}{}".format(
                        metric, prev, loss_val, f"{colcode}{sign}{abs(chng):.5f}", COL_RESET, f"{colcode}{sign}{100*abs(pct):.2f}%", COL_RESET))
                print("Perf: {:.6f}".format(self.optimise_perf / self.optimise_calls))
                print("Proc: {:.6f}".format(self.optimise_time / self.optimise_calls))
                self.prev_emas = {k : v for k, v in self.loss_emas.items()}
            
            self.live_emas = {k : v for k, v in self.loss_emas.items()}
            self.grad_vecs = {}
            
            if tb_logger is not None:
                for metric, loss_val in self.loss_emas.items():
                    tb_logger.add_scalar(f'emas/{metric}', loss_val, current_iter / self.accum_limit)
                #tb_logger.add_scalar(f'Scalers/Generator', self.scaler_g.get_scale(), current_iter / self.accum_limit)
                #tb_logger.add_scalar(f'Scalers/Discriminator', self.scaler_d.get_scale(), current_iter / self.accum_limit)
            
        if self.accum_count_d >= self.accum_limit and has_net_d:
            self.accum_count_d = 0
            self.grad_updates_d += 1
            # Discriminator
            # gradient clipping on discriminator
            if self.gradclip:
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), 0.01, error_if_nonfinite=False)
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
            self.optimizer_d.zero_grad(set_to_none=True)
            self.optim_to_sched[self.optimizer_d].step()

        # error if NaN
        #if torch.isnan(l_g_total):
        #    msg = """
        #          NaN found, aborting training. Make sure you're using a proper learning rate.
        #          If you have AMP enabled, try using bfloat16. For more information:
        #          https://github.com/muslll/neosr/wiki/Configuration-Walkthrough
        #          """
        #    raise ValueError(msg)
            
        #self.log_dict = self.reduce_loss_dict(loss_dict)
                
        for key, value in loss_dict.items():
            val = value if type(value) is float else value.item()
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples
            
        self.optimise_perf += time.perf_counter() - optimise_start_perf
        self.optimise_time += time.process_time() - optimise_start_time


    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warm-up iter numbers. -1 for no warm-up.
                Default： -1.
        """

        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def test(self):
        self.tile = self.opt['val'].get('tile', -1)
        if self.tile == -1:
            self.net_g.eval()
            with torch.inference_mode():
                self.output = self.net_g(self.lq)
            self.net_g.train()

        # test by partitioning
        else:
            _, C, h, w = self.lq.size()
            split_token_h = h // self.tile + 1  # number of horizontal cut sections
            split_token_w = w // self.tile + 1  # number of vertical cut sections

            patch_size_tmp_h = split_token_h
            patch_size_tmp_w = split_token_w
            
            # padding
            mod_pad_h, mod_pad_w = 0, 0
            if h % patch_size_tmp_h != 0:
                mod_pad_h = patch_size_tmp_h - h % patch_size_tmp_h
            if w % patch_size_tmp_w != 0:
                mod_pad_w = patch_size_tmp_w - w % patch_size_tmp_w

            img = self.lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h+mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w+mod_pad_w]

            _, _, H, W = img.size()
            split_h = H // split_token_h  # height of each partition
            split_w = W // split_token_w  # width of each partition

            # overlapping
            shave_h = 16
            shave_w = 16
            ral = H // split_h
            row = W // split_w
            slices = []  # list of partition borders
            for i in range(ral):
                for j in range(row):
                    if i == 0 and i == ral - 1:
                        top = slice(i * split_h, (i + 1) * split_h)
                    elif i == 0:
                        top = slice(i*split_h, (i+1)*split_h+shave_h)
                    elif i == ral - 1:
                        top = slice(i*split_h-shave_h, (i+1)*split_h)
                    else:
                        top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                    if j == 0 and j == row - 1:
                        left = slice(j*split_w, (j+1)*split_w)
                    elif j == 0:
                        left = slice(j*split_w, (j+1)*split_w+shave_w)
                    elif j == row - 1:
                        left = slice(j*split_w-shave_w, (j+1)*split_w)
                    else:
                        left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                    temp = (top, left)
                    slices.append(temp)
            img_chops = []  # list of partitions
            for temp in slices:
                top, left = temp
                img_chops.append(img[..., top, left])

            self.net_g.eval()
            with torch.inference_mode():
                outputs = []
                for chop in img_chops:
                    out = self.net_g(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * self.scale, W * self.scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * self.scale, (i + 1) * split_h * self.scale)
                        left = slice(j * split_w * self.scale, (j + 1) * split_w * self.scale)
                        if i == 0:
                            _top = slice(0, split_h * self.scale)
                        else:
                            _top = slice(shave_h * self.scale, (shave_h + split_h) * self.scale)
                        if j == 0:
                            _left = slice(0, split_w * self.scale)
                        else:
                            _left = slice(shave_w * self.scale, (shave_w + split_w) * self.scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * self.scale, 0:w - mod_pad_w * self.scale]

    @torch.no_grad()
    def feed_data(self, data):
        #self.lq = data['lq'].to(self.device, memory_format=torch.channels_last, non_blocking=True)
        self.lq = data['lq'].to(self.device, non_blocking=True)
        if 'gt' in data:
            #self.gt = data['gt'].to(self.device, memory_format=torch.channels_last, non_blocking=True)
            self.gt = data['gt'].to(self.device, non_blocking=True)

        # augmentation
        if self.is_train and self.aug is not None:
            self.gt, self.lq = apply_augment(self.gt, self.lq, scale=self.scale, augs=self.aug, prob=self.aug_prob)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img)

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):

        dataset_name = dataloader.dataset.opt['name']
        dataset_type = dataloader.dataset.opt['type']
        save_img     = dataloader.dataset.opt.get('save_img', False)
        save_tb      = dataloader.dataset.opt.get('save_tb', False)
        with_metrics = dataloader.dataset.opt.get('metrics', False)
        
        if not save_img and not save_tb and not with_metrics:
            return
            
        # flag to not apply augmentation during val
        self.is_train = False        
        
        #print(dataset_name, dataset_type, save_img, with_metrics)
        
        #if dataset_type == "single":
        #    with_metrics = False
        #else:
        #    with_metrics = self.opt['val'].get('metrics') is not None
            
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            
            # check if dataset has save_img option, and if so overwrite global save_img option
            #save_img = self.opt["val"].get("save_img", False)
            #print(dataset_name, dataset_type, save_img, with_metrics, img_name)
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if save_tb:
                tb_logger.add_image(f'{img_name}/{current_iter}', sr_img[:,:,[2,1,0]], global_step=current_iter, dataformats='HWC')
                
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        self.is_train = True

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'........ Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self, lq=True, result=True, gt=True):
        out_dict = OrderedDict()
        if lq:
            out_dict['lq'] = self.lq.detach().cpu()
        if result:
            out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            if gt:
                out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def validation(self, dataloader, current_iter, tb_logger, save_img=True):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def get_current_log(self):
        return {k : v / self.loss_samples for k, v in self.log_dict.items()}
        #return self.log_dict
    
    def reset_current_log(self):
        self.log_dict = {}
        self.loss_samples = 0

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        if self.opt.get('use_amp', False) is True:
            net = net.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        else:
            net = net.to(self.device, non_blocking=True)

        if self.opt['compile'] is True:
            net = torch.compile(net)
            # see option fullgraph=True

        if self.opt['dist']:
            find_unused_parameters = self.opt.get(
                'find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

        logger = get_root_logger()
        logger.info(
            f'Network: {net_cls_str}, with parameters: {net_params:,d}, and trainable: {trainable:,d}')
        logger.info(net_str)

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key, strict=True):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(
                    f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}.')
            raise IOError(f'Cannot save {save_path}.')

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        self.save_network(self.net_g, 'net_g', current_iter)

        if self.net_d is not None:
            self.save_network(self.net_d, 'net_d', current_iter)

        self.save_training_state(epoch, current_iter)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, param_key, strict=True):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: None.
        """
        self.param_key = param_key
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=torch.device('cuda'))

        try:
            if 'params-ema' in load_net:
                param_key = 'params-ema'
            elif 'params' in load_net:
                param_key = 'params'
            elif 'params_ema' in load_net:
                param_key = 'params_ema'
            else:
                param_key = self.param_key
            load_net = load_net[param_key]
        except:
            pass

        if param_key:
            logger.info(
                f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        else:
            logger.info(
                f'Loading {net.__class__.__name__} model from {load_path}.')

        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)
        torch.cuda.empty_cache()

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """

        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter,
                    'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{int(current_iter)}.state'
            save_path = os.path.join(
                self.opt['path']['training_states'], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(
                        f'Save training state error: {e}, remaining retry times: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(
                    f'Still cannot save {save_path}. Just ignore it.')
                raise IOError(f'Cannot save, aborting.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(
            self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(
            self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.inference_mode():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses, strict=True)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
