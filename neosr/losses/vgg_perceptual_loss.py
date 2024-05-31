from collections import OrderedDict

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from neosr.archs.vgg_arch import VGGFeatureExtractor
from neosr.losses.basic_loss import chc
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with VGG19

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        criterion (str): Criterion used for perceptual loss. Default: 'huber'.
    """

    def __init__(
        self,
        layer_weights: OrderedDict,
        vgg_type: str = "vgg19",
        use_input_norm: bool = True,
        range_norm: bool = False,
        perceptual_weight: float = 1.0,
        criterion: str = "huber",
        **kwargs,
    ) -> None:
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        elif self.criterion_type == "fro":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion not supported.")
            
        self.manual_wgts = {"conv5_4" : torch.from_numpy(np.array([0.1       , 0.1       , 0.05555556, 0.5       , 0.16666667,
       0.05555556, 0.2       , 0.2       , 0.16666667, 0.09090909,
       0.08333333, 0.33333333, 0.1       , 0.1       , 0.04545455,
       0.14285714, 0.33333333, 0.2       , 0.5       , 0.25      ,
       0.1       , 0.05555556, 0.04545455, 0.03448276, 0.06666667,
       0.1       , 0.04545455, 0.03448276, 0.05882353, 0.11111111,
       0.33333333, 0.1       , 0.16666667, 0.1       , 0.06666667,
       0.0625    , 0.06666667, 0.11111111, 0.16666667, 1.        ,
       0.16666667, 0.07142857, 0.03448276, 0.5       , 0.125     ,
       0.33333333, 0.33333333, 0.16666667, 0.06666667, 0.16666667,
       0.05555556, 0.5       , 0.0625    , 0.03448276, 0.06666667,
       0.05882353, 0.125     , 0.06666667, 0.2       , 0.0625    ,
       0.16666667, 0.07692308, 0.08333333, 0.14285714, 0.07142857,
       0.05882353, 0.08333333, 0.09090909, 0.1       , 0.06666667,
       0.09090909, 0.07142857, 0.04545455, 0.05882353, 0.2       ,
       0.5       , 0.2       , 0.06666667, 0.04545455, 0.07142857,
       0.1       , 0.08333333, 0.125     , 0.08333333, 0.03448276,
       0.08333333, 0.07142857, 0.14285714, 0.5       , 0.05555556,
       0.06666667, 0.0625    , 0.03448276, 0.03448276, 0.03448276,
       0.09090909, 0.25      , 0.09090909, 0.11111111, 0.03448276,
       0.125     , 0.08333333, 0.16666667, 0.03448276, 0.1       ,
       0.2       , 0.05555556, 0.09090909, 0.03448276, 0.07692308,
       0.0625    , 0.16666667, 0.06666667, 0.08333333, 0.03448276,
       0.1       , 0.1       , 0.06666667, 0.25      , 0.33333333,
       0.1       , 0.0625    , 0.08333333, 0.25      , 0.05882353,
       0.33333333, 0.0625    , 0.125     , 0.16666667, 0.03448276,
       0.07142857, 0.08333333, 0.09090909, 0.07142857, 0.09090909,
       0.09090909, 0.0625    , 0.11111111, 0.2       , 0.1       ,
       0.1       , 0.125     , 0.1       , 0.06666667, 0.1       ,
       0.03448276, 0.06666667, 0.16666667, 0.04545455, 0.07142857,
       0.06666667, 0.14285714, 0.16666667, 0.2       , 0.06666667,
       0.03448276, 0.07692308, 0.1       , 0.04545455, 0.03448276,
       0.0625    , 0.1       , 0.11111111, 0.06666667, 0.06666667,
       0.2       , 0.07142857, 0.25      , 0.05882353, 0.16666667,
       0.1       , 0.09090909, 0.08333333, 0.14285714, 0.04545455,
       0.0625    , 0.09090909, 0.05882353, 0.03448276, 0.16666667,
       0.05882353, 0.33333333, 0.1       , 0.16666667, 0.04545455,
       0.2       , 0.03448276, 0.05882353, 0.06666667, 0.09090909,
       0.0625    , 0.07692308, 0.07142857, 0.11111111, 0.16666667,
       0.16666667, 0.16666667, 0.125     , 0.1       , 0.33333333,
       0.03448276, 0.2       , 0.07142857, 0.1       , 1.        ,
       0.03448276, 0.1       , 0.125     , 0.2       , 0.16666667,
       0.05555556, 0.16666667, 0.125     , 0.07142857, 0.1       ,
       0.06666667, 0.14285714, 0.16666667, 0.06666667, 0.07142857,
       0.11111111, 0.05555556, 0.33333333, 0.16666667, 0.04545455,
       0.25      , 0.07692308, 0.03448276, 0.2       , 0.0625    ,
       0.16666667, 0.04545455, 0.05555556, 0.06666667, 0.1       ,
       0.04545455, 1.        , 0.16666667, 0.03448276, 0.2       ,
       0.04545455, 0.1       , 0.09090909, 0.0625    , 0.03448276,
       0.07142857, 0.09090909, 0.1       , 0.1       , 0.08333333,
       0.2       , 0.0625    , 0.1       , 0.2       , 0.07142857,
       0.08333333, 0.08333333, 0.1       , 0.06666667, 0.08333333,
       0.09090909, 0.05882353, 0.05555556, 0.04545455, 0.06666667,
       0.06666667, 0.1       , 0.07692308, 0.1       , 0.125     ,
       0.09090909, 0.06666667, 0.04545455, 0.08333333, 0.33333333,
       0.08333333, 0.1       , 0.125     , 0.33333333, 0.1       ,
       0.1       , 0.1       , 0.1       , 0.0625    , 0.08333333,
       0.03448276, 0.05555556, 0.04545455, 0.1       , 0.06666667,
       1.        , 0.08333333, 0.33333333, 0.08333333, 0.03448276,
       0.03448276, 0.08333333, 0.1       , 0.06666667, 0.03448276,
       0.08333333, 0.03448276, 0.2       , 0.0625    , 0.03448276,
       0.03448276, 0.09090909, 0.5       , 0.16666667, 0.06666667,
       0.05555556, 0.25      , 0.04545455, 0.05555556, 0.05882353,
       0.03448276, 0.11111111, 0.06666667, 0.25      , 0.0625    ,
       0.125     , 0.08333333, 0.08333333, 0.1       , 0.07142857,
       0.03448276, 0.03448276, 0.1       , 0.03448276, 0.06666667,
       0.14285714, 0.05882353, 0.5       , 0.1       , 0.08333333,
       0.03448276, 0.14285714, 0.07142857, 0.09090909, 0.04545455,
       0.04545455, 0.08333333, 0.33333333, 0.06666667, 0.16666667,
       0.33333333, 0.0625    , 0.25      , 0.06666667, 0.16666667,
       0.125     , 0.03448276, 0.06666667, 0.25      , 0.03448276,
       0.1       , 0.08333333, 0.05555556, 0.14285714, 0.05882353,
       0.07692308, 0.16666667, 0.14285714, 0.03448276, 0.0625    ,
       0.09090909, 0.03448276, 0.25      , 0.05882353, 0.07142857,
       0.09090909, 0.07692308, 0.03448276, 0.03448276, 0.08333333,
       0.16666667, 0.04545455, 0.125     , 0.14285714, 0.125     ,
       0.05555556, 0.07142857, 0.07692308, 0.0625    , 0.03448276,
       0.03448276, 0.07142857, 0.05555556, 0.1       , 0.08333333,
       0.03448276, 0.03448276, 0.03448276, 0.03448276, 0.08333333,
       0.16666667, 0.06666667, 0.03448276, 0.0625    , 0.08333333,
       0.07142857, 0.0625    , 0.06666667, 0.16666667, 0.16666667,
       0.08333333, 0.0625    , 0.2       , 0.14285714, 0.05882353,
       0.125     , 0.25      , 0.07142857, 0.06666667, 0.11111111,
       0.08333333, 0.14285714, 0.2       , 0.1       , 0.06666667,
       0.06666667, 0.0625    , 0.05882353, 0.05882353, 0.16666667,
       0.16666667, 0.07142857, 0.1       , 0.16666667, 0.2       ,
       0.33333333, 0.25      , 0.03448276, 0.09090909, 0.07692308,
       0.06666667, 0.03448276, 0.0625    , 0.1       , 0.06666667,
       0.03448276, 0.03448276, 0.0625    , 0.04545455, 0.0625    ,
       0.5       , 0.07692308, 0.1       , 0.5       , 0.1       ,
       0.1       , 0.33333333, 0.16666667, 0.1       , 0.5       ,
       1.        , 0.0625    , 0.07692308, 0.03448276, 0.16666667,
       0.0625    , 0.33333333, 0.0625    , 0.03448276, 0.06666667,
       0.25      , 0.08333333, 0.06666667, 0.16666667, 0.1       ,
       0.08333333, 0.2       , 0.2       , 0.03448276, 0.1       ,
       0.1       , 0.25      , 0.33333333, 0.06666667, 0.1       ,
       0.1       , 0.07142857, 0.03448276, 0.1       , 0.14285714,
       1.        , 0.07142857, 0.33333333, 0.2       , 0.0625    ,
       0.07142857, 0.09090909, 0.1       , 0.05555556, 0.5       ,
       0.03448276, 0.16666667, 0.05555556, 0.07692308, 0.0625    ,
       0.25      , 0.2       , 0.08333333, 0.33333333, 0.03448276,
       0.07142857, 0.04545455, 0.06666667, 0.07142857, 0.16666667,
       0.06666667, 0.16666667]))
       }
       
        for k, v in self.manual_wgts.items():
            self.manual_wgts[k] = v.unsqueeze(-1).unsqueeze(-1).cuda()

    def forward(
        self, x: torch.Tensor, gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features  = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        percep_loss = 0.0

        # calculate perceptual loss
        if self.perceptual_weight != 0:
            for k in x_features.keys():
                if self.layer_weights[k] == 0:
                    continue
                if False and k in self.manual_wgts:
                    #print(gt_features[k].shape)
                    spec = self.manual_wgts[k]#.unsqueeze(-1).unsqueeze(-1).cuda()
                    percep_loss += self.layer_weights[k] * self.criterion(x_features[k] * spec, gt_features[k] * spec)
                elif self.criterion_type == "fro":
                    # note: linalg.norm uses Frobenius norm by default
                    percep_loss += self.layer_weights[k] * torch.linalg.norm(x_features[k] - gt_features[k])
                else:
                    percep_loss += self.layer_weights[k] * self.criterion(x_features[k], gt_features[k])

        return percep_loss
