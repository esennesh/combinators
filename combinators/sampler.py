#!/usr/bin/env python3

import inspect
import torch

from .tracing import NestedTrace, TraceDiagram, TracedLensBox
from . import utils

class ImportanceSampler:
    def __init__(self, name, target, proposal, batch_shape=(1,)):
        super().__init__()
        assert callable(target) and hasattr(target, 'update')
        self._batch_shape = batch_shape
        self._name = name

        self.target = target
        sig = inspect.signature(target.forward)
        target_batching = 'batch_shape' in sig.parameters

        self.proposal = proposal
        proposal_batching = False
        if isinstance(self.proposal, torch.nn.Module):
            sig = inspect.signature(self.proposal.forward)
            proposal_batching = 'batch_shape' in sig.parameters
        elif callable(self.proposal):
            sig = inspect.signature(self.proposal)
            proposal_batching = 'batch_shape' in sig.parameters

        self._trace = None
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
        return self._cache(q, *args, **kwargs)

    def _score(self, q, *args, **kwargs):
        p = NestedTrace(q=q)
        result = self.target(p, *args, **kwargs)

        log_weight = p.conditioning_factor(self.batch_shape)
        assert log_weight.shape == self.batch_shape

        return TraceDiagram.BOX(result, log_weight, p, self._name)

    def __call__(self, *args, **kwargs):
        if args and isinstance(args[-1], dict):
            kwargs = {**args[-1], **kwargs}
            args = args[:-1]
        if self._pass_batch_shape:
            kwargs['batch_shape'] = self.batch_shape

        if self.trace:
            q = self.trace.box()[2]

            args, kwargs = self._expand_args(*args, **kwargs)
            self._trace = self._cache(q, *args, **kwargs)
            result, _, _, _ = self._trace.box()
        else:
            self._trace = self._sample(*args, **kwargs)
            result = self._trace.box()[0]

        return result

    @property
    def trace(self):
        return self._trace

    def clear(self):
        self._cache.clear()
        self._trace = None

    def update(self, *args, **kwargs):
        assert self.trace

        args, kwargs = self._expand_args(*args, **kwargs)
        _, log_weight, p, name = self.trace.box()
        q, p = utils.split_latent(p)

        result, q = self.target.update(q, *args, **kwargs)
        assert all(not q[k].observed for k in q)

        p = utils.join_traces(q, p)
        self._trace = TraceDiagram.BOX(None, log_weight, p, name)
        return result

def importance_box(name, target, proposal, batch_shape, dom, cod):
    sampler = ImportanceSampler(name, target, proposal, batch_shape)
    return TracedLensBox(name, dom, cod, sampler, sampler.update)

class VariationalSampler(ImportanceSampler):
    def __init__(self, name, target, proposal, mk_optimizer, batch_shape=(1,)):
        super().__init__(name, target, proposal, batch_shape)
        self._optimizer = mk_optimizer(list(self.parameters()))

    def update(self, *args, **kwargs):
        self._optimizer.step()

        return super().update(*args, **kwargs)

    def clear(self):
        self._optimizer.zero_grad()
        super().clear()
