#!/usr/bin/env python3

import logging
import math
import probtorch
import torch
import torch.nn.functional as F
from discopy import wiring

from .. import sampler, utils

def diagram_ess(diagram):
    _, log_weight = sampler.trace(diagram)
    result = effective_sample_size(log_weight, log_weight.shape[:-1])
    logging.info('Effective sample size (ESS) of %.4e based upon K=%d samples',
                 result, math.prod(log_weight.shape[:-1]))
    return result

def diagram_log_Z_hat(diagram):
    _, log_weight = sampler.trace(diagram)
    result = log_Z_hat(log_weight, log_weight.shape[:-1])
    logging.info('Estimated log evidence (log Z) %.8e based upon K=%d samples',
                 result, math.prod(log_weight.shape[:-1]))
    return result

def effective_sample_size(log_weights, particle_shape):
    log_normalized = utils.normalize_weights(log_weights, particle_shape, True)
    log_normalized, _ = utils.batch_collapse(log_normalized, particle_shape)
    ess = (-torch.logsumexp(2 * log_normalized, dim=0)).exp()
    return ess.mean(dim=0)

def log_Z_hat(log_weights, particle_shape):
    log_weights, _ = utils.batch_collapse(log_weights, particle_shape)
    return probtorch.util.log_mean_exp(log_weights, dim=0).mean(dim=0)
