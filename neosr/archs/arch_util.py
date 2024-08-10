import collections.abc
import math
import warnings
from itertools import repeat
from pathlib import Path

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm

from neosr.utils.options import parse_options


def net_opt():
    # initialize options parsing
    root_path = Path(__file__).parents[2]
    opt, args = parse_options(root_path, is_train=True)

    # set variable for scale factor and training phase
    # conditions needed due to convert.py

    if args.input is None:
        upscale = opt['scale']
        if 'train' in opt['datasets']:
            training = True
        else:
            training = False
    else:
        upscale = args.scale
        training = False

    return upscale, training

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)

    '''
    # old implementation
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    '''

    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        __, training = net_opt()
        self.training = training

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse
    
class PadAndMask(torch.nn.Module):

    def __init__(self, pad_size, pad_value, fill=False, include_mask=False):
    
        super(PadAndMask, self).__init__()
        self.pad_size    = pad_size
        self.pad_value   = pad_value
        self.fill        = fill
        self.include_mask = include_mask
        self.padder_x    = nn.ConstantPad2d(pad_size, pad_value)
        self.padder_mask = nn.ConstantPad2d(pad_size, 0)

    def forward(self, x):
        if self.pad_size == 0:
            out = x
        elif self.fill:
            
            out = self.padder_mask(x)
            out[:, :, 0:self.pad_size, :] = out[:, :, self.pad_size, :].unsqueeze(2)
            out[:, :, out.shape[2]-self.pad_size:, :] = out[:, :, out.shape[2]-self.pad_size-1, :].unsqueeze(2)
            
            out[:, :, :, 0:self.pad_size] = out[:, :, :, self.pad_size].unsqueeze(3)
            out[:, :, :, out.shape[3]-self.pad_size:] = out[:, :, :, out.shape[3]-self.pad_size-1].unsqueeze(3)
            
            if self.include_mask:
                ones = torch.ones_like(x[:,0,:,:]).unsqueeze(1)
                mask = 1 - self.padder_mask(ones)
                out  = torch.cat((out, mask), dim=1)
        else:
            out = self.padder_x(x)
            if self.include_mask:
                ones = torch.ones_like(x[:,0,:,:]).unsqueeze(1)
                mask = 1 - self.padder_mask(ones)
                out = torch.cat((out, mask), dim=1)
        return out
        
class TrainedPadder(torch.nn.Module):

    def __init__(self, channels_in, pad_size, kernel_size=3):
    
        super(TrainedPadder, self).__init__()
        self.channels_in = channels_in
        self.pad_size    = pad_size
        self.kernel_size = kernel_size
        self.corner_size = self.pad_size**2 + 2*self.pad_size*(self.kernel_size//2)
        
        self.conv_u = nn.Conv2d(self.channels_in, self.channels_in, self.kernel_size, stride=1)
        self.conv_d = nn.Conv2d(self.channels_in, self.channels_in, self.kernel_size, stride=1)
        self.conv_l = nn.Conv2d(self.channels_in, self.channels_in, self.kernel_size, stride=1)
        self.conv_r = nn.Conv2d(self.channels_in, self.channels_in, self.kernel_size, stride=1)
        
        self.conv_u.weight = torch.nn.Parameter((1.0 / self.kernel_size ** 2 / self.channels_in) * torch.ones_like(self.conv_u.weight))
        self.conv_d.weight = torch.nn.Parameter((1.0 / self.kernel_size ** 2 / self.channels_in) * torch.ones_like(self.conv_d.weight))
        self.conv_l.weight = torch.nn.Parameter((1.0 / self.kernel_size ** 2 / self.channels_in) * torch.ones_like(self.conv_l.weight))
        self.conv_r.weight = torch.nn.Parameter((1.0 / self.kernel_size ** 2 / self.channels_in) * torch.ones_like(self.conv_r.weight))
        
        self.conv_u.bias = torch.nn.Parameter(torch.zeros_like(self.conv_u.bias))
        self.conv_d.bias = torch.nn.Parameter(torch.zeros_like(self.conv_d.bias))
        self.conv_l.bias = torch.nn.Parameter(torch.zeros_like(self.conv_l.bias))
        self.conv_r.bias = torch.nn.Parameter(torch.zeros_like(self.conv_r.bias))
        
        self.conv_ul = nn.Conv2d(self.channels_in, self.channels_in, self.kernel_size, stride=1)
        self.conv_ur = nn.Conv2d(self.channels_in, self.channels_in, self.kernel_size, stride=1)
        self.conv_bl = nn.Conv2d(self.channels_in, self.channels_in, self.kernel_size, stride=1)
        self.conv_br = nn.Conv2d(self.channels_in, self.channels_in, self.kernel_size, stride=1)
        
         
        self.fc_in       = self.channels_in * (self.pad_size + self.kernel_size)**2
        self.fc_out      = self.channels_in * self.corner_size
        self.fc_grid     = round((self.fc_in / self.channels_in)**0.5)
        self.dense_ul    = nn.Linear(self.fc_in, self.fc_out, bias=False)
        self.dense_ur    = nn.Linear(self.fc_in, self.fc_out, bias=False)
        self.dense_bl    = nn.Linear(self.fc_in, self.fc_out, bias=False)
        self.dense_br    = nn.Linear(self.fc_in, self.fc_out, bias=False)
        self.zero_padder = nn.ConstantPad2d(pad_size, 0)

    def forward(self, x):
    
        if self.pad_size == 0:
            out = x
            
        else:
            
            p = self.pad_size
            k = self.kernel_size
            s = p + k // 2
            
            out      = self.zero_padder(x)
            filter_u = self.conv_u(x[:, :, :k + p - 1, :])
            filter_d = self.conv_d(x[:, :, -(k + p - 1): ,:])
            filter_l = self.conv_l(x[:, :, :, :k + p - 1])
            filter_r = self.conv_r(x[:, :, :, -(k + p - 1):])

            out[:, :, 0:p, s:out.shape[3]-s] = filter_u[:, :, :, :]
            out[:, :, -p:, s:out.shape[3]-s] = filter_d[:, :, :, :]
            out[:, :, s:out.shape[2]-s, 0:p] = filter_l[:, :, :, :]
            out[:, :, s:out.shape[2]-s, -p:] = filter_r[:, :, :, :]
            
            filter_ul = self.conv_ul(x[:, :, :k+s-1, :k+s-1])
            filter_ur = self.conv_ur(x[:, :, :k+s-1, -k-s+1:])
            filter_bl = self.conv_bl(x[:, :, -k-s+1:, :k+s-1])
            filter_br = self.conv_br(x[:, :, -k-s+1:, -k-s+1:])
            
            # UL
            out[:, :, :p, :s]    = filter_ul[:,:, :p, :s]
            out[:, :, p:s, :p]   = filter_ul[:,:, p:s, :p]
            # UR
            out[:, :, :p, -s:]   = filter_ur[:,:, :p, -s:]
            out[:, :, p:s, -p:]  = filter_ur[:,:, p:s, -p:]
            # BL
            out[:, :, -p:, :s]   = filter_bl[:,:, -p:, :s]
            out[:, :, -s:-p, :p] = filter_bl[:, :, -s:-p, :p]
            # BR
            out[:, :, -p:, -s:]  = filter_br[:,:, -p:, -s:]
            out[:, :, -s:-p, -p:] = filter_br[:, :, -s:-p, -p:]
            
            '''
            SZ     = self.fc_grid
            UL_pix = out[:, :, 0:SZ, 0:SZ]
            UR_pix = out[:, :, -SZ:, -SZ:]
            BL_pix = out[:, :, out.shape[2]-SZ:, 0:SZ]
            BR_pix = out[:, :, out.shape[2]-SZ:, out.shape[3]-SZ:]
            
            b      = x.shape[0]
            UL_out = self.dense_ul(UL_pix.reshape(b, self.fc_in)).view(b, self.channels_in, self.pad_size, -1)
            UR_out = self.dense_ur(UR_pix.reshape(b, self.fc_in)).view(b, self.channels_in, self.pad_size, -1)
            BL_out = self.dense_bl(BL_pix.reshape(b, self.fc_in)).view(b, self.channels_in, self.pad_size, -1)
            BR_out = self.dense_br(BR_pix.reshape(b, self.fc_in)).view(b, self.channels_in, self.pad_size, -1)
            
            out[:, :, 0:self.pad_size, 0:self.pad_size+offset]                 = UL_out[:, :, :, 0:(self.pad_size+offset)]
            out[:, :, self.pad_size:self.pad_size+offset, 0:self.pad_size]     = UL_out[:, :, :, (self.pad_size+offset):].permute(0,1,3,2)
            out[:, :, 0:self.pad_size, -(self.pad_size+offset):]               = UR_out[:, :, :, 0:(self.pad_size+offset)]
            out[:, :, self.pad_size:self.pad_size+offset, -(self.pad_size):]   = UR_out[:, :, :, (self.pad_size+offset):].permute(0,1,3,2)
            out[:, :, -self.pad_size:, 0:self.pad_size+offset]                 = BL_out[:, :, :, 0:(self.pad_size+offset)]
            out[:, :, -(self.pad_size+offset):-self.pad_size, 0:self.pad_size] = BL_out[:, :, :, (self.pad_size+offset):].permute(0,1,3,2)
            out[:, :, -self.pad_size:, -(self.pad_size+offset):]               = BR_out[:, :, :, 0:(self.pad_size+offset)]
            out[:, :, -(self.pad_size+offset):-self.pad_size, -self.pad_size:] = BR_out[:, :, :, (self.pad_size+offset):].permute(0,1,3,2)
            '''
                
        return out


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
