
import torch
from torch import nn
from torch.nn import functional as F

from neosr.utils.registry import ARCH_REGISTRY
from .arch_util import net_opt
from .arch_util import PadAndMask

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

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=96, num_head=1, num_body=12, num_tail=1, upscale=upscale, act_type='prelu', **kwargs):
        super(custom, self).__init__()
        self.num_in_ch  = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat   = num_feat
        self.num_body   = num_body
        self.num_head   = num_head
        self.num_tail   = num_tail
        self.upscale    = upscale
        self.act_type   = act_type
        self.pad_mode   = "zeros"
        PAD_CONSTANT    = 0.0
        self.pad_fill   = False
        self.pad_mask   = False
        
        self.head = nn.ModuleList()
        self.body = nn.ModuleList()
        self.tail = nn.ModuleList()
        self.end  = nn.ModuleList()
        
        def get_activation(num_channels):
            if self.act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif self.act_type == 'prelu':
                #activation = nn.PReLU(num_parameters=num_channels)
                activation = nn.PReLU(num_parameters=num_channels)
            elif self.act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            return activation
        
        
        # the first conv
        init_k       = 1
        n_feat_in    = num_in_ch + int(self.pad_mask)
        num_out_head = n_feat_in * 4
        padding_needed = 0
        
        for i in range(self.num_head):
            padding     = init_k // 2
            padding_needed += padding
            in_offset   = 0 if (not self.pad_mask or padding == 0) else 1
            #if padding > 0:
                #self.head.append(nn.ConstantPad2d(padding, PAD_CONSTANT))
                #self.head.append(PadAndMask(padding, PAD_CONSTANT, fill=self.pad_fill, include_mask=self.pad_mask))
            self.head.append(nn.Conv2d(n_feat_in, num_out_head, init_k))#, 1, init_k//2, padding_mode=self.pad_mode))
            # the first activation
            activation = get_activation(num_out_head)
            self.head.append(activation)
            n_feat_in = num_out_head

        #n_feat_in = num_out_head
        # the body structure
        for i in range(self.num_body):
            n_feat_next = n_feat_in
            k           = 3 if i % 2 == 0 else 3
            padding     = k // 2
            padding_needed += padding
            in_offset   = 0 if (not self.pad_mask or padding == 0) else 1
            #k = 11#2*(i % 4)+1
            #if padding > 0:
                #self.body.append(nn.ConstantPad2d(padding, PAD_CONSTANT))
                #self.body.append(PadAndMask(padding, PAD_CONSTANT, fill=self.pad_fill, include_mask=self.pad_mask))
            self.body.append(nn.Conv2d(n_feat_in, n_feat_next, k))#, 1, padding, padding_mode=self.pad_mode))
            # activation
            activation = get_activation(n_feat_next)
            #self.body.append(nn.BatchNorm2d(num_feat, momentum=0.10))
            self.body.append(activation)
            n_feat_in = n_feat_next
            
            #if i > 0 and i % 3 == 0 and (num_body - i) >= 0:
            #    self.body.append(nn.BatchNorm2d(num_feat, momentum=0.20))
            
        #n_feat_in += num_out_head + self.num_in_ch
        for i in range(self.num_tail):
            n_feat_next = max(num_out_ch * upscale ** 2 * 2, int(n_feat_in * 0.75))
            k           = 1
            padding     = k // 2
            padding_needed += padding
            in_offset   = 0 if (not self.pad_mask or padding == 0) else 1
            #if padding > 0:
                #self.tail.append(nn.ConstantPad2d(padding, PAD_CONSTANT))
                #self.tail.append(PadAndMask(padding, PAD_CONSTANT, fill=self.pad_fill, include_mask=self.pad_mask))
            self.tail.append(nn.Conv2d(n_feat_in, n_feat_next, k))#, 1, padding, padding_mode=self.pad_mode))
            
            # activation
            activation = get_activation(n_feat_next)
            self.tail.append(activation)
            n_feat_in = n_feat_next

        # the last conv
        #self.body.append(nn.Conv2d(num_feat, num_out_ch *
        #                 upscale * upscale, 3, 1, 1))
        
        # the last conv
        
        k           = 1
        padding     = k // 2
        padding_needed += padding
        in_offset   = 0 if (not self.pad_mask or padding == 0) else 1
        #if padding > 0:
            #self.end.append(PadAndMask(padding, PAD_CONSTANT, fill=self.pad_fill, include_mask=self.pad_mask))
        self.end.append(nn.Conv2d(n_feat_in, num_out_ch * self.upscale ** 2, k))
        #self.concat_conv = nn.Conv2d(n_feat_in + in_offset, num_out_ch * self.upscale ** 2, k)#, 1, 1, padding_mode=self.pad_mode)
        
        #self.padder = PadAndMask(padding_needed, 0.0, fill=self.pad_fill, include_mask=self.pad_mask)
        self.padder = None
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        
        #if upscale == 1:
        #    base = x
        #else:
        #    base = F.interpolate(x, scale_factor=self.upscale, mode='nearest-exact')
        
        #base = x
        #out  = x
        
        if self.training:
            out = x
        else:
        
            if self.padder is None:
            
                out = x
                # Head
                for i in range(0, len(self.head)):
                    out = self.head[i](out)
                head_out = out
                
                # Body            
                for i in range(0, len(self.body)):
                    out = self.body[i](out)
                    
                #out = torch.cat((out, head_out, base), 1)
                
                # Tail
                for i in range(0, len(self.tail)):
                    out = self.tail[i](out)
                    
                # End
                for i in range(0, len(self.end)):
                    out = self.end[i](out)
                    
                padding_needed = (x.shape[2] - out.shape[2]) // 2
                self.padder    = PadAndMask(padding_needed, 0.0, fill=True, include_mask=False)
                
            out = self.padder(x)
        
        #out = self.padder(x)
        
        # Head
        for i in range(0, len(self.head)):
            out = self.head[i](out)
        head_out = out
        
        # Body            
        for i in range(0, len(self.body)):
            out = self.body[i](out)
            
        #out = torch.cat((out, head_out, base), 1)
        
        # Tail
        for i in range(0, len(self.tail)):
            out = self.tail[i](out)
            
        # End
        for i in range(0, len(self.end)):
            out = self.end[i](out)
        
        #out = torch.cat((x, out), 1)
        #out = self.padder_last(out)
        #out = self.concat_conv(out)
        #out = self.last_active(out)
        #out = self.last_conv(out)

        out  = self.upsampler(out)
        #if upscale == 1:
        #    base = x
        #else:
        #    #base = F.interpolate(x, scale_factor=self.upscale, mode='nearest-exact')
        #    base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear')
            
        # add the nearest upsampled image, so that the network learns the residual
        #out += base
        
        return out
