#!/usr/bin/env python3

import torch.nn as nn

from .tracing import NestedTrace, TraceDiagram
from . import utils

class ImportanceSampler(nn.Module):
    def __init__(self, name, target, proposal, batch_shape=(1,)):
        super().__init__()
        assert callable(target) and hasattr(target, 'update')
        self._batch_shape = batch_shape
        self._name = name

        self.add_module('target', target)
        if isinstance(proposal, nn.Module):
            self.add_module('proposal', proposal)
        else:
            self.proposal = None

        self._cache = utils.TensorialCache(1, self._sample)

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
        return self._score(q, *args, **kwargs)

    def _score(self, q, *args, **kwargs):
        p = NestedTrace(q=q)
        result = self.target(p, *args, **kwargs)

        log_weight = p.conditioning_factor(self.batch_shape)
        assert log_weight.shape == self.batch_shape

        return TraceDiagram.BOX(result, log_weight, p, self._name)

    def forward(self, *args, **kwargs):
        if args and isinstance(args[-1], dict):
            kwargs = args[-1]
            args = args[:-1]

        result, _, p, _ = self._cache(*args, **kwargs).box()
        if result is None:
            args, kwargs = self._expand_args(*args, **kwargs)
            self._cache[0] = ((args, kwargs), self._score(p, *args, **kwargs))
            result = self.trace.box()[0]

        return result

    @property
    def trace(self):
        if self._cache:
            return self._cache[0][-1]
        return None

    def clear(self):
        self._cache.clear()

    def update(self, *args, **kwargs):
        assert self.trace

        args, kwargs = self._expand_args(*args, **kwargs)
        _, log_weight, p, name = self.trace.box()
        q, p = utils.split_latent(p)

        if hasattr(self.proposal, 'update'):
            self.proposal.update(q, *args, **kwargs)
        result, q = self.target.update(q, *args, **kwargs)
        assert all(not q[k].observed for k in q)

        p = utils.join_traces(q, p)
        trace = TraceDiagram.BOX(None, log_weight, p, name)
        self._cache[0] = (self._cache[0][0], trace)
        return result

class VariationalSampler(ImportanceSampler):
    def __init__(self, name, target, proposal, mk_optimizer, batch_shape=(1,)):
        super().__init__(name, target, proposal, batch_shape)
        self._optimizer = mk_optimizer(list(self.parameters()))

    def update(self, *args, **kwargs):
        self._optimizer.step()
        self._optimizer.zero_grad()
        return super().update(*args, **kwargs)
