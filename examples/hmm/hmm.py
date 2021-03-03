#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.functional import softplus

import combinators.model as model
import combinators.utils as utils
from examples.gmm.gmm import GaussianClusters

class Parameters(nn.Module):
    def __init__(self, num_states, dim=2, state_params={}):
        super().__init__()
        self._dim = dim
        self._num_states = num_states

        self.states = GaussianClusters(num_states, dim=dim, **state_params)
        self.register_parameter('pi', nn.Parameter(torch.ones(num_states)))

    def forward(self, p, batch_shape=(1,)):
        mus, sigmas = self.states(p, batch_shape=batch_shape)
        pi = utils.batch_expand(self.pi, batch_shape)
        pi = torch.stack([p.dirichlet(pi, name='pi_%d' % (k+1)) for k
                          in range(self._num_states)], dim=-1)
        z0 = p.variable(Categorical, probs=pi[:, 0], name='z_0')
        return mus, sigmas, pi, z0

    def update(self, p):
        return (), p

class TransitionAndEmission(nn.Module):
    def forward(self, p, mus, sigmas, pi, z, data=None):
        pis = utils.particle_index(pi, z)
        zs = p.variable(Categorical, probs=pis, name='z')

        mu = utils.particle_index(mus, z)
        sigma = utils.particle_index(sigmas, z)
        p.normal(mu, sigma, name='x', value=data)

        return mus, sigmas, pi, zs

    def update(self, p):
        return (), p
