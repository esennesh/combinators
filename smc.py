#!/usr/bin/env python3

import probtorch
from probtorch.stochastic import RandomVariable, Trace
import torch
from torch.nn.functional import log_softmax

import combinators
import utils

class ParticleTrace(probtorch.stochastic.Trace):
    def __init__(self, num_particles=1):
        super(ParticleTrace, self).__init__()
        self._num_particles = num_particles

    @property
    def num_particles(self):
        return self._num_particles

    def variable(self, Dist, *args, **kwargs):
        args = [arg.expand(self.num_particles, *arg.shape)
                if isinstance(arg, torch.Tensor) and
                arg.shape[0] != self.num_particles
                else arg for arg in args]
        kwargs = {k: v.expand(self.num_particles, *v.shape)
                     if isinstance(v, torch.Tensor) and
                     v.shape[0] != self.num_particles
                     else v for k, v in kwargs.items()}
        return super(ParticleTrace, self).variable(Dist, *args, **kwargs)

    def log_joint(self, *args, **kwargs):
        return super(ParticleTrace, self).log_joint(*args, sample_dim=0,
                                                    **kwargs)

    def resample(self, weights):
        resampler = torch.distributions.Categorical(logits=weights)
        particles = resampler.sample((self.num_particles,))

        result = ParticleTrace(self.num_particles)
        for i, key in enumerate(self.variables()):
            rv = self[key]
            if not rv.observed:
                value = rv.value.index_select(0, particles)
                sample = RandomVariable(rv.dist, value, rv.observed, rv.mask,
                                        rv.reparameterized)
            else:
                sample = rv
            if key is not None:
                result[key] = sample
            else:
                result[i] = sample

        return result

def likelihood_weight(trace):
    rvs = [rv for rv in trace.variables() if trace[rv].observed][-1]
    rvs = [rvs]
    return log_softmax(trace.log_joint(reparameterized=False, nodes=rvs), dim=0)
