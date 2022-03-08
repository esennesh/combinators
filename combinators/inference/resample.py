#!/usr/bin/env python3

from abc import abstractmethod
from discopy import cartesian, monoidal
from functools import wraps
from probtorch import RandomVariable
import torch
import torch.nn.functional as F

from .. import lens, sampler, utils

def collapsed_index_select(tensor, ancestors):
    particle_shape = ancestors.shape
    tensor, unique = utils.batch_collapse(tensor, particle_shape)
    ancestors, _ = utils.batch_collapse(ancestors, particle_shape)
    tensor = tensor.index_select(0, ancestors)
    return tensor.reshape(particle_shape + unique)

def index_select_rv(rv, ancestors):
    result = rv
    if isinstance(rv, RandomVariable) and not rv.observed:
        value = collapsed_index_select(rv.value, ancestors)
        result = RandomVariable(rv.Dist, value, *rv.dist_args,
                                provenance=rv.provenance, mask=rv.mask,
                                **rv.dist_kwargs)
    return result

class Resampler:
    def __init__(self, diagram, particle_shape):
        self._diagram = diagram
        self._particle_shape = particle_shape

    @property
    def diagram(self):
        return self._diagram

    @abstractmethod
    def ancestor_indices(self, log_weight):
        pass

    @staticmethod
    def resample_sampler(sampler, ancestors, particle_shape):
        (args, kwargs), (results, p, lw) = sampler.peek()
        args, results = list(args), list(results)
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                args[i] = collapsed_index_select(arg, ancestors)
        for k, v in kwargs.items():
            if k != 'data' and torch.is_tensor(v):
                kwargs[k] = collapsed_index_select(v, ancestors)
        for i, result in enumerate(results):
            if torch.is_tensor(result):
                results[i] = collapsed_index_select(result, ancestors)

        index = lambda rv: index_select_rv(rv, ancestors)
        p = utils.trace_map(p, index)
        if torch.is_tensor(lw):
            lw = utils.batch_mean(lw, particle_shape).expand(*ancestors.shape)

        args, results = tuple(args), tuple(results)
        sampler.cache[(args, kwargs)] = (results, p, lw)

    def resample_diagram(self, *args, **kwargs):
        _, log_weight = sampler.trace(self.diagram)
        if not torch.is_tensor(log_weight) or (log_weight == 0.).all():
            return args, kwargs
        ancestors = self.ancestor_indices(log_weight)
        assert ancestors.shape == log_weight.shape

        for box in self.diagram:
            if isinstance(box, sampler.ImportanceWiringBox) and\
               box.sampler.cache:
                self.resample_sampler(box.sampler, ancestors,
                                      self._particle_shape)

        args = list(args)
        for i, val in enumerate(args):
            if torch.is_tensor(val):
                args[i] = collapsed_index_select(val, ancestors)
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                kwargs[k] = collapsed_index_select(v, ancestors)
        return tuple(args), kwargs

class SystematicResampler(Resampler):
    def ancestor_indices(self, log_weight):
        log_weights, unique = utils.batch_collapse(log_weight,
                                                   self._particle_shape)
        weights = F.softmax(log_weights, dim=0)
        K = self._particle_shape[0]

        uniforms = torch.rand(*unique, device=weights.device)
        uniforms = uniforms.expand(K, *unique).transpose(0, 1)

        positions = torch.arange(K, device=weights.device).expand(*unique, K)
        positions = ((positions + uniforms) / K).transpose(0, 1)

        cumsums = torch.cumsum(weights, dim=0)
        (normalizers, _) = torch.max(input=cumsums, dim=0, keepdim=True)
        normalized_cumsums = cumsums / normalizers

        indices = torch.searchsorted(normalized_cumsums, positions)
        assert indices.shape == (K, *unique)
        return indices

class MultinomialResampler(Resampler):
    def ancestor_indices(self, log_weight):
        log_weights, unique = utils.batch_collapse(log_weight,
                                                   self._particle_shape)
        weights = F.softmax(log_weights, dim=0).unsqueeze(dim=0)
        indices = torch.multinomial(weights, weights.shape[1], replacement=True)
        return indices.reshape(*log_weight.shape, *unique)

def hook_resampling(graph, particle_shape, method='get',
                    resampler_cls=SystematicResampler, when='post'):
    resampler = resampler_cls(graph, particle_shape)

    for box in graph:
        if isinstance(box, sampler.ImportanceWiringBox):
            kwargs = {when + '_' + method: resampler.resample_diagram}
            lens.hook(box, **kwargs)
