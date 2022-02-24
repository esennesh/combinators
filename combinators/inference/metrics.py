#!/usr/bin/env python3

import probtorch
import torch
import torch.nn.functional as F

from .. import utils

def effective_sample_size(log_weights, particle_shape):
    log_normalized = utils.normalize_weights(log_weights, particle_shape, True)
    log_normalized, _ = utils.batch_collapse(log_normalized, particle_shape)
    ess = (-torch.logsumexp(2 * log_normalized, dim=0)).exp()
    return ess.mean(dim=0)

def log_Z_hat(log_weights, particle_shape):
    log_weights, _ = utils.batch_collapse(log_weights, particle_shape)
    return probtorch.util.log_mean_exp(log_weights, dim=0).mean(dim=0)
