#!/usr/bin/env python3

import probtorch
from torch.distributions import Categorical, Normal
import torch
import torch.nn as nn

from combinators.tracing import TraceDiagram
from combinators.utils import batch_expand, particle_index

class GaussianClusters(nn.Module):
    def __init__(self, num_clusters, dim=2):
        super().__init__()

        self._num_clusters = num_clusters
        self._dim = dim

        self.register_buffer('mu', torch.zeros(1, self._num_clusters,
                                               self._dim))
        self.register_buffer('concentration', torch.ones(1, self._num_clusters,
                                                         self._dim) * 0.9)
        self.register_buffer('rate',
                             torch.ones(1, self._num_clusters, self._dim) * 0.9)

    def forward(self, p, batch_shape=(1,)):
        concentration = batch_expand(self.concentration, batch_shape)
        rate = batch_expand(self.rate, batch_shape)
        taus = p.gamma(concentration, rate, name='tau')
        sigmas = (1. / taus).sqrt()
        mu = batch_expand(self.mu, batch_shape)
        mus = p.normal(mu, sigmas, name='mu')

        return mus, sigmas

    def update(self, p, zs, xs):
        q = probtorch.Trace()

        zsk = nn.functional.one_hot(zs, self._num_clusters).unsqueeze(-1)
        xsk = xs.unsqueeze(2).expand(xs.shape[0], xs.shape[1],
                                     self._num_clusters, xs.shape[2]) * zsk
        nks = torch.stack([(zs == k).sum(dim=-1) + 1 for k in
                           range(self._num_clusters)], dim=-1).unsqueeze(-1)
        sample_means = xsk.sum(dim=1) / nks
        sample_sqdevs = (xsk - zsk * sample_means.unsqueeze(1)) ** 2

        concentration = self.concentration + nks / 2
        rate = self.rate + sample_sqdevs.sum(dim=1) / 2 +\
               nks * sample_means ** 2 / (2 * (nks + 1))
        taus = q.gamma(concentration, rate, name='tau')

        mean_tau = taus * (nks + 1)
        mean_mu = nks * sample_means / (1 + nks)
        q.normal(mean_mu, (1. / mean_tau).sqrt(), name='mu')

        return (), q

class SampleCluster(nn.Module):
    def __init__(self, num_clusters, num_samples):
        super().__init__()

        self._num_clusters = num_clusters
        self._num_samples = num_samples
        self.register_buffer('pi', torch.ones(self._num_samples,
                                              self._num_clusters))

    def forward(self, p, mus, sigmas):
        pi = self.pi.expand(mus.shape[0], *self.pi.shape)
        z = p.variable(Categorical, pi, name='z')
        return particle_index(mus, z), particle_index(sigmas, z)

    def update(self, p, mus, sigmas, xs):
        def log_likelihood(k):
            return Normal(mus[:, k], sigmas[:, k]).log_prob(xs).sum(-1)
        log_conditionals = torch.stack([log_likelihood(k) for k
                                        in range(mus.shape[1])], dim=-1)

        q = probtorch.Trace()
        z = q.variable(Categorical, logits=log_conditionals, name='z')
        return (z, xs), q

class SamplePoint(nn.Module):
    def forward(self, p, mu, sigma, x_observed=None):
        x = p.normal(mu, sigma, name='x', value=x_observed)
        return x

    def update(self, p, mu, sigma, xs):
        return xs, p
