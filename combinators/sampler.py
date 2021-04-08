#!/usr/bin/env python3

import inspect
import torch

from .tracing import NestedTrace, TracedLensBox
from . import utils

class ImportanceSampler:
    def __init__(self, target, proposal, batch_shape=(1,)):
        super().__init__()
        assert callable(target) and hasattr(target, 'update')
        self._batch_shape = batch_shape

        self.target = target
        sig = inspect.signature(target.forward)
        target_batching = 'batch_shape' in sig.parameters
        self._pass_data = 'data' in sig.parameters

        self.proposal = proposal
        proposal_batching = False
        if isinstance(self.proposal, torch.nn.Module):
            sig = inspect.signature(self.proposal.forward)
            proposal_batching = 'batch_shape' in sig.parameters
        elif callable(self.proposal):
            sig = inspect.signature(self.proposal)
            proposal_batching = 'batch_shape' in sig.parameters

        self._cache = utils.TensorialCache(1, self._score)
        self._pass_batch_shape = target_batching or proposal_batching

    @property
    def batch_shape(self):
        return self._batch_shape

    def _expand_args(self, *args, **kwargs):
        args = tuple(utils.batch_expand(arg, self.batch_shape, True)
                     if hasattr(arg, 'expand') else arg for arg in args)
        kwargs = {k: utils.batch_expand(v, self.batch_shape, True)
                     if hasattr(v, 'expand') else v for k, v in kwargs.items()}
        return args, kwargs

    def _sample(self, *args, **kwargs):
        args, kwargs = self._expand_args(*args, **kwargs)
        q = NestedTrace()

        if self.proposal is not None:
            self.proposal(q, *args, **kwargs)
        result, log_weight, p = self._score(q, *args, **kwargs)
        args = (p,) + args
        self._cache[(args, kwargs)] = (result, log_weight, p)
        return result, log_weight, p

    def _score(self, q, *args, **kwargs):
        p = NestedTrace(q=q)
        result = self.target(p, *args, **kwargs)

        log_weight = p.conditioning_factor(self.batch_shape)
        assert log_weight.shape == self.batch_shape

        return result, log_weight, p

    def sample(self, q, *args, **kwargs):
        if args and isinstance(args[-1], dict):
            kwargs = {**args[-1], **kwargs}
            args = args[:-1]
        if self._pass_batch_shape:
            kwargs['batch_shape'] = self.batch_shape
        if not self._pass_data and 'data' in kwargs:
            del kwargs['data']

        if q:
            args, kwargs = self._expand_args(*args, **kwargs)
            return self._cache(q, *args, **kwargs)
        return self._sample(*args, **kwargs)

    def clear(self):
        self._cache.clear()

    def update(self, q, *args, **kwargs):
        if not self._pass_data and 'data' in kwargs:
            del kwargs['data']

        args, kwargs = self._expand_args(*args, **kwargs)
        return self.target.update(q, *args, **kwargs)

def importance_box(name, target, proposal, batch_shape, dom, cod, data=None):
    sampler = ImportanceSampler(target, proposal, batch_shape)
    return TracedLensBox(name, dom, cod, sampler.sample, sampler.update,
                         data=data)

class VariationalSampler(ImportanceSampler):
    def __init__(self, target, proposal, mk_optimizer, batch_shape=(1,),
                 data=None):
        super().__init__(target, proposal, batch_shape, data)
        self._optimizer = mk_optimizer(list(target.parameters()) +\
                                       list(proposal.parameters()))

    def update(self, *args, **kwargs):
        self._optimizer.step()

        return super().update(*args, **kwargs)

    def clear(self):
        self._optimizer.zero_grad()
        super().clear()
