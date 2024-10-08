import torch
from torch import nn

from neosr.losses.basic_loss import chc, wgan
from neosr.utils.registry import LOSS_REGISTRY
#from torchvision.transforms.v2 import GaussianNoise

@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan' (l2) and 'huber'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(
        self,
        gan_type="vanilla",
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=0.1,
    ):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        
        self.losses_bcelogit = nn.BCEWithLogitsLoss()
        self.losses_bce = nn.BCELoss()
        self.losses_mse = nn.MSELoss()
        self.losses_huber = nn.HuberLoss()
        self.losses_chc = chc()

        if self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "nsgan":
            self.loss = None
        elif self.gan_type == "rsgan":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "bce":
            self.loss = nn.BCELoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "huber":
            self.loss = nn.HuberLoss()
        elif self.gan_type == "chc":
            self.loss = chc()
        elif self.gan_type == "wgan":
            self.loss = wgan()
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")
            
        #self.noise_layer = GaussianNoise(mean=0.0, sigma=0.005, clip=True)

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type == "wgan":
            target_val = -1.0 if target_is_real else 1.0
        else:
            target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, x, target_is_real, is_disc=False):
        """
        Args:
            x (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        
        x_mu = x.mean()
        if self.gan_type == "vanilla":
            if is_disc:
                loss = self.losses_bcelogit(x_mu/1e1, torch.ones_like(x_mu) * (0.9 if target_is_real else 0.0))
            else:
                loss = self.losses_bcelogit(x_mu/1e1, torch.ones_like(x_mu) * (1.0 if target_is_real else 0.0))
        
        elif self.gan_type == "lsgan":
            if is_disc:
                loss = self.losses_mse(torch.sigmoid(x_mu/1e1), torch.ones_like(x_mu) * (1.0 if target_is_real else 0.0))
            else:
                loss = self.losses_mse(torch.sigmoid(x_mu/1e1), torch.ones_like(x_mu) * (1.0 if target_is_real else 0.0))
                
        elif self.gan_type == "nsgan":
            if is_disc:
                if target_is_real:
                    loss = -torch.mean(torch.log(torch.sigmoid(x_mu/1e1) + 1e-7))
                else:
                    loss = -torch.mean(torch.log(1 - torch.sigmoid(x_mu/1e1) + 1e-7))
            else:
                loss = -torch.mean(torch.log(torch.sigmoid(x_mu/1e1) + 1e-7))
                
        elif self.gan_type == "wgan":
            if is_disc:
                if target_is_real:
                    loss = -x_mu
                else:
                    loss = x_mu
            else:
                loss = -x_mu
                
        else:
            target_label = self.get_target_label(x, target_is_real)
            loss = self.loss(x, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(
        self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0
    ):
        super(MultiScaleGANLoss, self).__init__(
            gan_type, real_label_val, fake_label_val, loss_weight
        )

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operation: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)
