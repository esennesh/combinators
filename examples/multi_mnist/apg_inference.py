#!/usr/bin/env python3

import logging
import torch

from combinators import sampler, signal
import combinators.inference.resample as resample
import combinators.inference.variational as variational

class ApgLoss(variational.VariationalLoss):
    def __init__(self, diagram):
        super().__init__(diagram)
        self._elbos = []
        self._eubos = []

    @property
    def elbos(self):
        return self._elbos

    @property
    def eubos(self):
        return self._eubos

    def objective(self):
        diagram = self._globals[0]
        _, log_weight = sampler.trace(diagram)

        theta_elbo = -variational.elbo(log_weight, iwae=True)
        self._elbos.append(theta_elbo)

        phi_eubo   = variational.eubo(log_weight, iwae=False)
        self._eubos.append(phi_eubo)

        return theta_elbo + phi_eubo

def hooks_apg(graph):
    resample.hook_resampling(graph, method='put', when='pre',
                             resampler_cls=resample.SystematicResampler)
    losses = ApgLoss(graph)
    variational.hook_variational(graph, losses, method='put', when='post')
    return losses
