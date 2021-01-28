#!/usr/bin/env python3

from torch.distributions import Categorical
import torch
import torch.nn as nn

class GaussianClusters(nn.Module):
    def __init__(self, num_clusters, dim=2):
        super().__init__()

        self._num_clusters = num_clusters
        self._dim = dim

        self.register_buffer('mu', torch.zeros(self._num_clusters, self._dim))
        self.register_buffer('concentration', torch.ones(self._num_clusters,
                                                         self._dim) * 0.9)
        self.register_buffer('rate',
                             torch.ones(self._num_clusters, self._dim) * 0.9)

    def forward(self, p):
        taus = p.gamma(self.concentration, self.rate, name='tau')
        sigmas = (1. / taus).sqrt()
        mus = p.normal(self.mu, sigmas, name='mu')

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

        concentration = self.concentration.unsqueeze(0) + nks / 2
        rate = self.rate.unsqueeze(0) +\
               sample_sqdevs.sum(dim=1) / 2 +\
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
        self.register_buffer('pi', torch.ones(self._num_clusters))

    def forward(self, p, mus, sigmas):
        pi = self.pi.expand(self._num_samples, self._num_clusters)
        z = p.variable(Categorical, pi, name='z')
        return mus[:, z], sigmas[:, z]

class SamplePoint(nn.Module):
    def forward(self, p, mu, sigma, x_observed=None):
        x = p.normal(mu, sigma, name='x', value=x_observed)
        return x
