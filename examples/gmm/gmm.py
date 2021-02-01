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
                                                         self._dim))
        self.register_buffer('rate', torch.ones(self._num_clusters, self._dim))

    def forward(self, p):
        mus = p.normal(self.mu, torch.ones_like(self.mu), name='mu')
        taus = p.gamma(self.concentration, self.rate, name='tau')

        return mus, 1. / taus

class SampleCluster(nn.Module):
    def __init__(self, num_clusters):
        super().__init__()

        self._num_clusters = num_clusters
        self.register_buffer('pi', torch.ones(self._num_clusters))

    def forward(self, p, mus, sigmas):
        z = p.variable(Categorical, self.pi, name='z')
        return (mus[:, z], sigmas[:, z]), p

class SamplePoint(nn.Module):
    def forward(self, p, mu, sigma, x_observed=None):
        x = p.normal(mu, sigma, name='x', value=x_observed)
        return x, p
