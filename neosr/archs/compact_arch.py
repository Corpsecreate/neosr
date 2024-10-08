
from torch import nn
from torch.nn import functional as F

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import net_opt

upscale, training = net_opt()


@ARCH_REGISTRY.register()
class compact(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=upscale, act_type='prelu', **kwargs):
    
        super(compact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        self.pad_mode = "zeros"
        
        #self.num_feat *= self.upscale
        #self.num_conv *= self.upscale

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(self.num_in_ch, self.num_feat, 3, 1, 1, padding_mode = self.pad_mode))
        # the first activation
        if self.act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif self.act_type == 'prelu':
            activation = nn.PReLU(num_parameters=self.num_feat)
        elif self.act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(self.num_conv):
            self.body.append(nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1, padding_mode = self.pad_mode))
            # activation
            if self.act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif self.act_type == 'prelu':
                activation = nn.PReLU(num_parameters=self.num_feat)
            elif self.act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(self.num_feat, self.num_out_ch * self.upscale ** 2, 3, 1, 1, padding_mode = self.pad_mode))
        # upsample
        self.upsampler = nn.PixelShuffle(self.upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
            
        if self.upscale != 1:
            out = self.upsampler(out)
            
        # add the nearest upsampled image, so that the network learns the residual
        base = x if self.upscale == 1 else F.interpolate(x, scale_factor=self.upscale, mode='nearest-exact')
        out += base
        
        return out
        
@ARCH_REGISTRY.register()
def compact_tiny(**kwargs):
    return compact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=8,
            num_conv=4,
            upscale=upscale,
            act_type='prelu',
            **kwargs
            )

@ARCH_REGISTRY.register()
def compact_small(**kwargs):
    return compact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=24,
            num_conv=8,
            upscale=upscale,
            act_type='prelu',
            **kwargs
            )

@ARCH_REGISTRY.register()
def compact_large(**kwargs):
    return compact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=96,
            num_conv=24,
            upscale=upscale,
            act_type='prelu',
            **kwargs
            )
            
@ARCH_REGISTRY.register()
class compact_gan(nn.Module):
    """A compact VGG-style network structure for super-resolution.

    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=upscale, act_type='prelu', **kwargs):
    
        super(compact_gan, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type
        self.pad_mode = "zeros"
        
        #self.num_feat *= self.upscale
        #self.num_conv *= self.upscale

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(self.num_in_ch, self.num_feat, 3, 1, 1, padding_mode = self.pad_mode))
        # the first activation
        if self.act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif self.act_type == 'prelu':
            activation = nn.PReLU(num_parameters=self.num_feat)
        elif self.act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(self.num_conv):
            self.body.append(nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1, padding_mode = self.pad_mode))
            # activation
            if self.act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif self.act_type == 'prelu':
                activation = nn.PReLU(num_parameters=self.num_feat)
            elif self.act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(self.num_feat, 1, 3, 1, 1, padding_mode = self.pad_mode))
        self.body.append(nn.Sigmoid())
        # upsample
        self.upsampler = nn.PixelShuffle(self.upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)
        
        return out