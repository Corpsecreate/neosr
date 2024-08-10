
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import net_opt
from .arch_util import PadAndMask, TrainedPadder

upscale, training = net_opt()

@ARCH_REGISTRY.register()
class custom(nn.Module):
    """A custom VGG-style network structure for super-resolution.

    It is a custom network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_body (int): Number of convolution layers in the body network. Default: 16.
        num_tail (int): Number of convolution layers in the tail network. Default: 4.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=128, num_conv=6, upscale=upscale, act_type='prelu', **kwargs):
    
        super(custom, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        self.pad_mode = "zeros"
        self.pad_fill = False
        self.pad_mask = False
        
        self.normalize = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                              std =[0.250, 0.250, 0.250])
        
        def get_activation(num_channels):
            if self.act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif self.act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_channels)
            elif self.act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            return activation
        
        self.total_padding = 0

        self.body = nn.ModuleList()
        # the first conv
        #self.body.append(PadAndMask(1, 0.0, fill=self.pad_fill, include_mask=self.pad_mask))
        
        k = 1; self.total_padding += k // 2
        nf_in  = self.num_in_ch
        nf_out = self.num_feat
        #if k//2 > 0: self.body.append(nn.ConstantPad2d(k//2, 0.0))
        if k//2 > 0: self.body.append(TrainedPadder(nf_in, k//2))
        self.body.append(nn.Conv2d(nf_in + self.pad_mask, nf_out, k, 1, 0, padding_mode = self.pad_mode))
        # the first activation
        self.body.append(get_activation(nf_out))
        
        nf_in = nf_out

        # the body structure
        for i in range(self.num_conv):
            #self.body.append(PadAndMask(1, 0.0, fill=self.pad_fill, include_mask=self.pad_mask))
            k = 5; self.total_padding += k // 2
            nf_out = nf_in# * (2 if i > 0 and i % 2 == 0 else 1)
            #if k//2 > 0: self.body.append(nn.ConstantPad2d(k//2, 0.0))
            if k//2 > 0: self.body.append(TrainedPadder(nf_in, k//2))
            self.body.append(nn.Conv2d(nf_in + self.pad_mask, nf_out, k, 1, 0, padding_mode = self.pad_mode))
            # activation
            self.body.append(get_activation(nf_out))
            nf_in = nf_out

        # the last conv
        #self.body.append(PadAndMask(1, 0.0, fill=self.pad_fill, include_mask=self.pad_mask))
        k = 1; self.total_padding += k // 2
        #if k//2 > 0: self.body.append(nn.ConstantPad2d(k//2, 0.0))
        if k//2 > 0: self.body.append(TrainedPadder(nf_in, k//2))
        self.body.append(nn.Conv2d(nf_in + self.pad_mask, self.num_out_ch * self.upscale ** 2, k, 1, 0, padding_mode = self.pad_mode))
        # upsample
        self.upsampler = nn.PixelShuffle(self.upscale)
        # padder at inference
        self.padder = PadAndMask(self.total_padding, 0.0, True, self.pad_mask)

    def forward(self, x):
    
        #interp = x if self.upscale == 1 else F.interpolate(x, scale_factor=self.upscale, mode='bilinear')
        if True:#self.training:
            out = x
        else:
            out = self.padder(x)
            
        out = self.normalize(x)
        
        for i in range(0, len(self.body)):
            out = self.body[i](out)
            
        if self.upscale != 1:
            out = self.upsampler(out)
            
        # add the nearest upsampled image, so that the network learns the residual
        base   = x if self.upscale == 1 else F.interpolate(x, scale_factor=self.upscale, mode='nearest-exact')
        w_diff = base.shape[2] - out.shape[2]
        h_diff = base.shape[3] - out.shape[3]
        out = out + base[:, :, 
                      w_diff//2 : base.shape[2] - w_diff//2,
                      h_diff//2 : base.shape[3] - h_diff//2]
        
        return out#torch.clamp(out, 0, 1)
        
@ARCH_REGISTRY.register()
class custom_compact(nn.Module):
    """A custom VGG-style network structure for super-resolution.

    It is a custom network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_body (int): Number of convolution layers in the body network. Default: 16.
        num_tail (int): Number of convolution layers in the tail network. Default: 4.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=80, num_conv=10, upscale=upscale, act_type='prelu', **kwargs):
    
        super(custom_compact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        self.pad_mode = "zeros"
        self.pad_fill = False
        self.pad_mask = False
        
        def get_activation(num_channels):
            if self.act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif self.act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_channels)
            elif self.act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            return activation
        
        self.total_padding = 0

        self.body = nn.ModuleList()
        # the first conv
        #self.body.append(PadAndMask(1, 0.0, fill=self.pad_fill, include_mask=self.pad_mask))
        k = 1; self.total_padding += k // 2
        nf_in  = self.num_in_ch
        nf_out = self.num_feat
        self.body.append(nn.Conv2d(nf_in + self.pad_mask, nf_out, k, 1, 0, padding_mode = self.pad_mode))
        # the first activation
        self.body.append(get_activation(nf_out))
        
        nf_in = nf_out

        # the body structure
        for i in range(self.num_conv):
            #self.body.append(PadAndMask(1, 0.0, fill=self.pad_fill, include_mask=self.pad_mask))
            k = 5; self.total_padding += k // 2
            nf_out = nf_in# * (2 if i > 0 and i % 2 == 0 else 1)
            self.body.append(nn.Conv2d(nf_in + self.pad_mask, nf_out, k, 1, 0, padding_mode = self.pad_mode))
            # activation
            self.body.append(get_activation(nf_out))
            nf_in = nf_out

        # the last conv
        #self.body.append(PadAndMask(1, 0.0, fill=self.pad_fill, include_mask=self.pad_mask))
        k = 1; self.total_padding += k // 2
        self.body.append(nn.Conv2d(nf_in + self.pad_mask, self.num_out_ch * self.upscale ** 2, k, 1, 0, padding_mode = self.pad_mode))
        # upsample
        self.upsampler = nn.PixelShuffle(self.upscale)
        # padder at inference
        self.padder = PadAndMask(self.total_padding, 0.0, True, self.pad_mask)

    def forward(self, x):
    
        #interp = x if self.upscale == 1 else F.interpolate(x, scale_factor=self.upscale, mode='bilinear')
        if self.training:
            out = x
        else:
            out = self.padder(x)
            
        for i in range(0, len(self.body)):
            out = self.body[i](out)
            
        if self.upscale != 1:
            out = self.upsampler(out)
            
        # add the nearest upsampled image, so that the network learns the residual
        base = x if self.upscale == 1 else F.interpolate(x, scale_factor=self.upscale, mode='nearest-exact')
        w_diff = base.shape[2] - out.shape[2]
        h_diff = base.shape[3] - out.shape[3]
        #out += base
        out += base[:, 
                    :, 
                    w_diff//2 : base.shape[2] - w_diff//2,
                    h_diff//2 : base.shape[3] - h_diff//2]
        
        return torch.clamp(out, 0, 1)
