from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms

from neosr.utils.registry import ARCH_REGISTRY
from neosr.archs.arch_util import net_opt

upscale, __ = net_opt()


class DCCM(nn.Sequential):
    "Doubled Convolutional Channel Mixer"

    def __init__(self, dim: int):
        super().__init__(
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Mish(),
            nn.Conv2d(dim * 2, dim, 3, 1, 1),
        )
        trunc_normal_(self[-1].weight, std=0.02)


class PLKConv2d(nn.Module):
    "Partial Large Kernel Convolutional Layer"

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2)
        trunc_normal_(self.conv.weight, std=0.02)
        self.idx = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
            x1 = self.conv(x1)
            return torch.cat([x1, x2], dim=1)
        x[:, : self.idx] = self.conv(x[:, : self.idx])
        return x


class EA(nn.Module):
    "Element-wise Attention"

    def __init__(self, dim: int):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1), nn.Sigmoid())
        trunc_normal_(self.f[0].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.f(x)


class PLKBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int,
        split_ratio: float,
        norm_groups: int,
        use_ea: bool = True,
    ):
        super().__init__()

        # Local Texture
        self.channel_mixer = DCCM(dim)

        # Long-range Dependency
        pdim = int(dim * split_ratio)

        # Conv Layer
        self.lk = PLKConv2d(pdim, kernel_size)

        # Instance-dependent modulation
        if use_ea:
            self.attn = EA(dim)
        else:
            self.attn = nn.Identity()

        # Refinement
        self.refine = nn.Conv2d(dim, dim, 1, 1, 0)
        trunc_normal_(self.refine.weight, std=0.02)

        # Group Normalization
        self.norm = nn.GroupNorm(norm_groups, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        x = self.channel_mixer(x)
        x = self.lk(x)
        x = self.attn(x)
        x = self.refine(x)
        x = self.norm(x)

        return x + x_skip


@ARCH_REGISTRY.register()
class realplksr(nn.Module):
    """Partial Large Kernel CNNs for Efficient Super-Resolution:
    https://arxiv.org/abs/2404.11848
    """

    def __init__(
        self,
        dim: int = 64,
        n_blocks: int = 28,
        upscaling_factor: int = upscale,
        kernel_size: int = 17,
        split_ratio: float = 0.25,
        use_ea: bool = True,
        norm_groups: int = 4,
        dropout: float = 0,
        **kwargs,
    ):
        super().__init__()
        
        self.upscaling_factor = upscaling_factor
        self.normalize = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                              std =[0.250, 0.250, 0.250])

        if not self.training:
            dropout = 0

        self.feats = nn.Sequential(
            *[nn.Conv2d(3, dim, 3, 1, 1)]
            + [
                PLKBlock(dim, kernel_size, split_ratio, norm_groups, use_ea)
                for _ in range(n_blocks)
            ]
            + [nn.Dropout2d(dropout)]
            + [nn.Conv2d(dim, 3 * self.upscaling_factor**2, 3, 1, 1)]
        )
        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)

        self.repeat_op = partial(
            torch.repeat_interleave, repeats=self.upscaling_factor**2, dim=1
        )

        self.upscaler = nn.PixelShuffle(self.upscaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        x_normed = self.normalize(x)
        out      = self.feats(x_normed) + self.repeat_op(x_normed)
        
        if self.upscaling_factor != 1:
            out = self.upscaler(out)
        
        base   = x if self.upscaling_factor == 1 else F.interpolate(x, scale_factor=self.upscaling_factor, mode='nearest-exact')
        w_diff = base.shape[2] - out.shape[2]
        h_diff = base.shape[3] - out.shape[3]
        out   += base[:, :, w_diff//2 : base.shape[2] - w_diff//2, h_diff//2 : base.shape[3] - h_diff//2]

        return torch.clamp(out, 0, 1)


@ARCH_REGISTRY.register()
def realplksr_s(**kwargs):
    return realplksr(n_blocks=12, kernel_size=13, use_ea=False, **kwargs)