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

    def forward(self, p, batch_shape=(1,), particle_shape=(1,)):
        shape = particle_shape + batch_shape
        concentration = particle_expand(self.concentration, shape)
        rate = particle_expand(self.rate, shape)
        taus = p.gamma(concentration, rate, name='tau')
        sigmas = (1. / taus).sqrt()
        mu = particle_expand(self.mu, shape)
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
        mu = self.mu.expand(*zsk.shape[:2], *self.mu.shape)
        concentration = self.concentration.expand(*zsk.shape[:2],
                                                  *self.concentration.shape)
        rate = self.rate.expand(*zsk.shape[:2], *self.rate.shape)

        nks = zsk.sum(dim=2)
        eff_samples = nks + 1
        hyper_means = (mu + xsk.sum(dim=2)) / eff_samples
        concentration = concentration + nks / 2
        rate = rate + 1/2 * (mu ** 2 - eff_samples * hyper_means ** 2 +
                             (xsk ** 2).sum(dim=2))

        precisions = q.gamma(concentration, rate, name='tau') * eff_samples
        q.normal(hyper_means, torch.pow(precisions, -1/2.), name='mu')

    def feedback(self, p, zsk, xsk):
        return ()

class SampleCluster(nn.Module):
    def __init__(self, num_clusters, num_observations):
        super().__init__()

        self._num_clusters = num_clusters
        self._num_observations = num_observations
        self.register_buffer('pi', torch.ones(self._num_clusters))

    def forward(self, p, batch_shape=(1,), particle_shape=(1,)):
        pi = particle_expand(self.pi, particle_shape + batch_shape +
                             (self._num_observations,))
        return p.variable(Categorical, pi, name='z')

class AssignmentGibbs(nn.Module):
    def forward(self, q, log_conditionals):
        z = q.variable(Categorical, logits=log_conditionals, name='z')

    def feedback(self, p, log_conditionals):
        return ()

class SamplePoint(nn.Module):
    def forward(self, p, mus, sigmas, z, data=None):
        z = z.unsqueeze(-1)
        mu = torch.take_along_dim(mus, z , dim=2)
        sigma = torch.take_along_dim(sigmas, z , dim=2)
        x = p.normal(mu, sigma, name='x', value=data)
        return x

class ObservationGibbs(nn.Module):
    def forward(self, q, mus, sigmas, zs, xs, data=None):
        pass

    def feedback(self, p, mus, sigmas, zs, data=None):
        xs = data.unsqueeze(dim=1)
        num_clusters = mus.shape[2]
        def log_likelihood(k):
            normal = Normal(mus[:, :, k], sigmas[:, :, k])
            return normal.log_prob(xs).sum(dim=-1)
        log_conditionals = torch.stack([log_likelihood(k) for k
                                        in range(num_clusters)], dim=-1)

        zsk = nn.functional.one_hot(zs, num_clusters).unsqueeze(-1)
        xsk = xs.unsqueeze(3).expand(*xs.shape[:3], num_clusters, *xs.shape[3:])
        xsk = xsk * zsk
        return (zsk, xsk, log_conditionals)
