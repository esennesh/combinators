#!/usr/bin/env python3

import probtorch
from torch.distributions import Categorical, Normal
import torch
import torch.nn as nn

from combinators.utils import particle_expand, particle_index

class GaussianClusters(nn.Module):
    def __init__(self, num_clusters, dim=2, mu=None, concentration=None,
                 rate=None):
        super().__init__()

        self._num_clusters = num_clusters
        self._dim = dim

        if mu is None:
            mu = torch.zeros(self._num_clusters, self._dim)
        if concentration is None:
            concentration = torch.ones(self._num_clusters, self._dim) * 0.9
        if rate is None:
            rate = torch.ones(self._num_clusters, self._dim) * 0.9

        self.register_buffer('mu', mu)
        self.register_buffer('concentration', concentration)
        self.register_buffer('rate', rate)

    def forward(self, p, particle_shape=(1,)):
        concentration = particle_expand(self.concentration, particle_shape)
        rate = particle_expand(self.rate, particle_shape)
        taus = p.gamma(concentration, rate, name='tau')
        sigmas = (1. / taus).sqrt()
        mu = particle_expand(self.mu, particle_shape)
        mus = p.normal(mu, sigmas, name='mu')

        return mus, sigmas

class ClustersGibbs(nn.Module):
    def __init__(self, num_clusters, dim=2, mu=None, concentration=None,
                 rate=None):
        super().__init__()
        self._num_clusters = num_clusters
        self._dim = dim

        if mu is None:
            mu = torch.zeros(self._num_clusters, self._dim)
        if concentration is None:
            concentration = torch.ones(self._num_clusters, self._dim) * 0.9
        if rate is None:
            rate = torch.ones(self._num_clusters, self._dim) * 0.9

        self.register_buffer('mu', mu)
        self.register_buffer('concentration', concentration)
        self.register_buffer('rate', rate)

    def forward(self, q, zsk, xsk):
        nks = zsk.sum(dim=1)
        eff_samples = nks + 1
        hyper_means = (self.mu.unsqueeze(0) + xsk.sum(dim=1)) / eff_samples
        concentration = self.concentration.unsqueeze(0) + nks / 2
        rate = self.rate.unsqueeze(0)
        rate = rate + 1/2 * (self.mu.unsqueeze(0) ** 2 -
                             eff_samples * hyper_means ** 2 +
                             (xsk ** 2).sum(dim=1))

        precisions = q.gamma(concentration, rate, name='tau') * eff_samples
        q.normal(hyper_means, torch.pow(precisions, -1/2.), name='mu')

    def feedback(self, p, zsk, xsk):
        return ()

class SampleCluster(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()

        self._num_clusters = num_clusters
        self.register_buffer('pi', torch.ones(self._num_clusters))

    def forward(self, p, batch_shape=(1,), particle_shape=(1,)):
        pi = particle_expand(self.pi, particle_shape + batch_shape)
        return p.variable(Categorical, pi, name='z')

class AssignmentGibbs(nn.Module):
    def forward(self, q, log_conditionals):
        z = q.variable(Categorical, logits=log_conditionals, name='z')

    def feedback(self, p, log_conditionals):
        return ()

class SamplePoint(nn.Module):
    def forward(self, p, mus, sigmas, z, data=None):
        mu, sigma = particle_index(mus, z), particle_index(sigmas, z)
        x = p.normal(mu, sigma, name='x', value=data)
        return ()

class ObservationGibbs(nn.Module):
    def forward(self, q, mus, sigmas, zs, data=None):
        pass

    def feedback(self, p, mus, sigmas, zs, data=None):
        xs = data
        num_clusters = mus.shape[1]
        def log_likelihood(k):
            return Normal(mus[:, k], sigmas[:, k]).log_prob(xs).sum(dim=-1)
        log_conditionals = torch.stack([log_likelihood(k) for k
                                        in range(num_clusters)], dim=-1)

        zsk = nn.functional.one_hot(zs, num_clusters).unsqueeze(-1)
        xsk = xs.unsqueeze(2).expand(xs.shape[0], xs.shape[1], num_clusters,
                                     xs.shape[2]) * zsk
        return (zsk, xsk, log_conditionals)
