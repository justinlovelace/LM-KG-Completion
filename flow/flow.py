import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import scipy
import scipy.linalg

class InvConv(nn.Module):
    """Invertible 1x1 Convolution for 2D inputs. Originally described in Glow
    (https://arxiv.org/abs/1807.03039). Does not support LU-decomposed version.

    Args:
        num_channels (int): Number of channels in the input and output.
    Adapted from: https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
    """
    def __init__(self, num_channels, lu=True):
        super(InvConv, self).__init__()
        self.num_channels = num_channels
        # Initialize with a random orthogonal matrix
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not lu:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.p = nn.Parameter(torch.Tensor(np_p.astype(np.float32)), requires_grad=False)
            self.sign_s = nn.Parameter(torch.Tensor(np_sign_s.astype(np.float32)), requires_grad=False)
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = nn.Parameter(torch.Tensor(l_mask), requires_grad=False)
            self.eye = nn.Parameter(torch.Tensor(eye), requires_grad=False)
        self.w_shape = w_shape
        self.lu = lu
            
    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.lu:
            dlogdet = torch.linalg.slogdet(self.weight)[1]
            if not reverse:
                weight = self.weight.view(self.num_channels, self.num_channels)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(self.num_channels, self.num_channels)
            return weight, dlogdet
        else:
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = torch.sum(self.log_s)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(self.num_channels, self.num_channels), dlogdet


    def forward(self, x, sldj, reverse=False):
        """
        log-det = log|abs(|W|)|
        """
        weight, ldj = self.get_weight(x, reverse)
        if not reverse:
            sldj = sldj + ldj
        else:
            sldj = sldj - ldj

        z = F.linear(x, weight)

        return z, sldj


class LinearFlow(nn.Module):
    """Glow Model

    Based on the paper:
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    by Diederik P. Kingma, Prafulla Dhariwal
    (https://arxiv.org/abs/1807.03039).

    Args:
        num_channels (int): Number of channels in middle convolution of each
            step of flow.
        num_levels (int): Number of levels in the entire model.
        num_steps (int): Number of steps of flow for each level.
    """
    def __init__(self, args, in_channels, lu=True):
        super(LinearFlow, self).__init__()

        self.map = InvConv(in_channels, lu=lu)

        self.bias = nn.Parameter(torch.zeros(in_channels), requires_grad=True)

    def forward(self, x, reverse=False):
        sldj = torch.zeros(x.size(0), device=x.device)

        # x = squeeze(x)
        x, sldj = self.map(x, sldj, reverse)
        x += self.bias.expand_as(x)

        return x, sldj
