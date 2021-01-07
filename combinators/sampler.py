#!/usr/bin/env python3

import torch.nn as nn

from .tracing import NestedTrace
from . import utils

class ImportanceSampler(nn.Module):
    def __init__(self, target, proposal, batch_shape=(1,)):
        super().__init__()
        self._batch_shape = batch_shape
        self.add_module('target', target)
        if isinstance(proposal, nn.Module):
            self.add_module('proposal', proposal)
        else:
            self.proposal = None

    @property
    def batch_shape(self):
        return self._batch_shape

    def forward(self, *args, **kwargs):
        if args and isinstance(args[-1], dict):
            kwargs = args[-1]
            args = args[:-1]
        args = tuple(arg.expand(*self._batch_shape, *arg.size())
                     if hasattr(arg, 'expand') else arg for arg in args)
        kwargs = {k: v.expand(*self._batch_shape, *v.size())
                     if hasattr(v, 'expand') else v for k, v in kwargs.items()}

        if self.proposal is not None:
            q = NestedTrace()
            _, q = self.proposal(q, *args, **kwargs)
            p = NestedTrace(q=q)
        else:
            p = NestedTrace()

        result, p = self.target(p, *args, **kwargs)

        log_weight = p.conditioning_factor(self.batch_shape)
        assert log_weight.shape == self.batch_shape
        return result, (log_weight, p)

    def update(self, args, feedback):
        raise NotImplementedError()

class VariationalSampler(ImportanceSampler):
    def __init__(self, target, proposal, mk_optimizer, batch_shape=(1,)):
        super().__init__(target, proposal, batch_shape)
        self._optimizer = mk_optimizer(list(self.parameters()))

    def forward(self, *args, **kwargs):
        self._optimizer.zero_grad()
        return super().forward(*args, **kwargs)

    def update(self, *args):
        trace = args[-1]
        self._optimizer.step()
        return trace
