#!/usr/bin/env python3

import collections
import logging

import numpy as np
import probtorch
import torch

from .inference import Inference
from . import importance
from ..kernel.kernel import TransitionKernel
from ..model import foldable
from ..sampler import Sampler
from .. import utils

class MarkovChain(Inference):
    def __init__(self, target, kernel, moves=1):
        super(MarkovChain, self).__init__(target)
        assert isinstance(kernel, TransitionKernel)
        self.add_module('kernel', kernel)
        self._moves = moves

    def forward(self, *args, **kwargs):
        zs, xi, log_weight = self.target(*args, **kwargs)
        multiple_zs = isinstance(zs, tuple)
        if not multiple_zs:
            zs = (zs,)

        for _ in range(self._moves):
            xiq = self.kernel(zs, xi, log_weight, *args, **kwargs)
            zsq, xiq, log_weight_q = self.target.cond(xiq)(*args, **kwargs)
            if self._accept(xi, log_weight, xiq, log_weight_q):
                zs = zsq
                xi = xiq
                log_weight = log_weight_q

        if not multiple_zs:
            zs = zs[0]
        return zs, xi, log_weight

class MHMove(MarkovChain):
    def _accept(self, xi, log_w, xiq, log_wq):
        marginal = utils.marginalize_all(log_w)
        marginal_q = utils.marginalize_all(log_wq)
        move_current = self.kernel.log_transition_prob(xiq, xi)
        move_candidate = self.kernel.log_transition_prob(xi, xiq)

        mh_ratio = (marginal_q - move_candidate) - (marginal - move_current)
        log_alpha = torch.min(torch.zeros(mh_ratio.shape), mh_ratio)
        return torch.bernoulli(torch.exp(log_alpha)) == 1

    def walk(self, f):
        return f(MHMove(self.target.walk(f), self.kernel, self._moves))

    def cond(self, qs):
        return MHMove(self.target.cond(qs), self.kernel, self._moves)

def mh_move(target, kernel, moves=1):
    return MHMove(target, kernel, moves=moves)

class LightweightKernel(TransitionKernel):
    def __init__(self, prior):
        super(LightweightKernel, self).__init__(prior.batch_shape)
        assert isinstance(prior, Sampler)
        self.add_module('prior', prior)

    def forward(self, zs, xi, log_weight, *args, **kwargs):
        sampled = []
        while not sampled:
            t = np.random.randint(len(xi))
            sampled = [k for k in xi[t].variables() if not xi[t][k].observed]

        key = sampled[np.random.randint(len(sampled))]
        candidate_trace = utils.slice_trace(xi[t], key)
        candidate_graph = xi.graft(t, candidate_trace)

        zsq, xiq, log_weight_q = self.prior.cond(candidate_graph)(*args,
                                                                  **kwargs)

        predicate = lambda k, v: not v.observed
        move_candidate = utils.marginalize_all(xiq.log_joint())
        move_candidate += torch.log(torch.tensor(float(
            xiq.num_variables(predicate=predicate)
        )))
        move_current = utils.marginalize_all(xi.log_joint())
        move_current += torch.log(torch.tensor(float(
            xi.num_variables(predicate=predicate)
        )))

        return zsq, xiq, log_weight_q, move_candidate, move_current

    def walk(self, f):
        return f(LightweightKernel(self.prior.walk(f)))

    def cond(self, qs):
        return LightweightKernel(self.target.cond(qs[self.name:]))

    @property
    def name(self):
        return 'LightweightKernel(%s)' % self.prior.name

def lightweight_mh(target, moves=1):
    return mh_move(target, LightweightKernel(target), moves=moves)

def resample_move_smc(target, moves=1, mcmc=lightweight_mh):
    inference = lambda m: mcmc(importance.Resample(m), moves)
    selector = lambda m: isinstance(m, importance.Importance)
    return target.apply(inference, selector)

def step_resample_move_smc(sampler, initializer=None, moves=1,
                           mcmc=lightweight_mh):
    return foldable.Step(mcmc(importance.Resample(sampler), moves),
                         initializer=initializer)

def reduce_resample_move_smc(stepwise, step_generator, initializer=None,
                             moves=1, mcmc=lightweight_mh):
    rmsmc_foldable = step_resample_move_smc(stepwise, initializer=initializer,
                                            moves=moves, mcmc=mcmc)
    return foldable.Reduce(rmsmc_foldable, step_generator)
