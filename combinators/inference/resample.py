#!/usr/bin/env python3

from abc import abstractmethod
from discopy import cartesian, monoidal
from functools import wraps
from probtorch import RandomVariable
import torch
import torch.nn.functional as F

from .. import lens, sampler, utils

def collapsed_index_select(tensor, batch_shape, ancestors):
    tensor, unique = utils.batch_collapse(tensor, batch_shape)
    tensor = tensor.index_select(0, ancestors)
    return tensor.reshape(batch_shape + unique)

def index_select_rv(rv, batch_shape, ancestors):
    result = rv
    if isinstance(rv, RandomVariable) and not rv.observed:
        value = collapsed_index_select(rv.value, batch_shape, ancestors)
        result = RandomVariable(rv.Dist, value, *rv.dist_args,
                                provenance=rv.provenance, mask=rv.mask,
                                **rv.dist_kwargs)
    return result

class Resampler:
    def __init__(self, diagram):
        self._diagram = diagram

    @property
    def diagram(self):
        return self._diagram

    @abstractmethod
    def ancestor_indices(self, log_weight):
        pass

    @staticmethod
    def resample_box(box, ancestors, batch_shape):
        (args, kwargs), (results, p, lw) = box.peek()
        args, results = list(args), list(results)
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                args[i] = collapsed_index_select(arg, batch_shape,
                                                 ancestors)
        for k, v in kwargs.items():
            if k != 'data' and torch.is_tensor(v):
                kwargs[k] = collapsed_index_select(v, batch_shape,
                                                   ancestors)
        for i, result in enumerate(results):
            if torch.is_tensor(result):
                results[i] = collapsed_index_select(result, batch_shape,
                                                    ancestors)

        index = lambda rv: index_select_rv(rv, batch_shape, ancestors)
        p = utils.trace_map(p, index)
        if torch.is_tensor(lw):
            lw = utils.batch_mean(lw, batch_shape).expand(*batch_shape)

        box.cache[(args, kwargs)] = (results, p, lw)

    def resample_diagram(self, *vals):
        _, log_weight = sampler.trace(self.diagram)
        if not torch.is_tensor(log_weight) or (log_weight == 0.).all():
            return vals
        ancestors = self.ancestor_indices(log_weight)
        batch_shape = log_weight.shape

        for box in self.diagram:
            if isinstance(box, sampler.ImportanceWiringBox) and box.cache:
                self.resample_box(box, ancestors, batch_shape)

        vals = list(vals)
        for i, val in enumerate(vals):
            if torch.is_tensor(val):
                vals[i] = collapsed_index_select(val, batch_shape, ancestors)
        return tuple(vals)

class SystematicResampler(Resampler):
    def ancestor_indices(self, log_weight):
        log_weights, _ = utils.batch_collapse(log_weight, log_weight.shape)
        weights = F.softmax(log_weights, dim=0)
        K = weights.shape[0]

        positions = torch.arange(K, device=weights.device) +\
                    torch.rand(1, device=weights.device)
        positions = positions / K

        indices = torch.zeros(K, device=weights.device, dtype=torch.long)
        cumulative_sum = torch.cumsum(weights, dim=0)
        i, j = 0, 0
        while i < K:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices.reshape(*log_weight.shape)

class MultinomialResampler(Resampler):
    def ancestor_indices(self, log_weight):
        log_weights, _ = utils.batch_collapse(log_weight, log_weight.shape)
        weights = F.softmax(log_weights, dim=0).unsqueeze(dim=0)
        indices = torch.multinomial(weights, weights.shape[1], replacement=True)
        return indices.reshape(*log_weight.shape)

def hook_resampling(graph, method='get', resampler_cls=SystematicResampler,
                    when='post'):
    resampler = resampler_cls(graph)

    for box in graph:
        if isinstance(box, sampler.ImportanceWiringBox):
            kwargs = {when + '_' + method: resampler.resample_diagram}
            lens.hook(box, **kwargs)
