import torch
from torch import nn
import numpy as np

from neosr.losses.basic_loss import chc
from neosr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class sobelloss(nn.Module):
    """Sobel Loss.
    https://automaticaddison.com/how-the-sobel-operator-works/

    Args:
        criterion (str): loss type. Default: 'huber'
        loss_weight (float): weight for colorloss. Default: 1.0
    """

    def __init__(
        self,
        criterion: str = "huber",
        loss_weight: float = 1.0,
    ) -> None:
        super(sobelloss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion_type = criterion

        if self.criterion_type.upper() in {"L1"}:
            self.criterion = nn.L1Loss()
        elif self.criterion_type.upper() in {"L2", "MSE"}:
            self.criterion = nn.MSELoss()
        elif self.criterion_type.upper() in {"HUBER"}:
            self.criterion = nn.HuberLoss()
        elif self.criterion_type.upper() in {"CHC"}:
            self.criterion = chc()
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")
            
        self.sobel_x = nn.Conv2d(1, 1, 3, 1, "valid", bias=False).cuda()
        self.sobel_x.weight = torch.nn.Parameter(torch.from_numpy(np.array([[[[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]]])), requires_grad=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, 1, "valid", bias=False).cuda()
        self.sobel_y.weight = torch.nn.Parameter(torch.from_numpy(np.array([[[[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]]]])), requires_grad=False)
        
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
        input_x  = self.sobel_x(input.view(-1, 1, input.shape[2], input.shape[3]))
        target_x = self.sobel_x(target.view(-1, 1, target.shape[2], target.shape[3]))
        loss_x   = self.criterion(input_x, target_x)
        
        input_y  = self.sobel_y(input.view(-1, 1, input.shape[2], input.shape[3]))
        target_y = self.sobel_y(target.view(-1, 1, target.shape[2], target.shape[3]))
        loss_y   = self.criterion(input_y, target_y)
        
        return loss_x + loss_y
