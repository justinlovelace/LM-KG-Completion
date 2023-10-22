import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.distributions as D
import sys


def bits_per_dim(x, nll):
    """Get the bits per dimension implied by using model with `loss`
    for compressing `x`, assuming each entry can take on `k` discrete values.

    Args:
        x (torch.Tensor): Input to the model. Just used for dimensions.
        nll (torch.Tensor): Scalar negative log-likelihood loss tensor.

    Returns:
        bpd (torch.Tensor): Bits per dimension implied if compressing `x`.
    """
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)

    return bpd


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)



class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.

    Args:
        k (int or float): Number of dimensions.
            E.g., `k` is 768 for BERT-Base models.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, args, k=768):
        super(NLLLoss, self).__init__()
        self.k = k
        self.args = args
        loc = torch.zeros(k, device='cuda')
        scale = torch.ones(k, device='cuda')
        self.distr = D.normal.Normal(loc=loc, scale=scale)

    def forward(self, z, sldj):
        d = {}
        prior_ll = self.distr.log_prob(z).sum(-1)
        ll = prior_ll + sldj
        nll = -ll.mean()
        d['prior_ll'] = prior_ll.mean()
        d['sldj'] = sldj.mean()
            
        d['loss'] = nll

        return d
