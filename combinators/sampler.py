#!/usr/bin/env python3

import torch.nn as nn

from .tracing import NestedTrace
from . import utils

class ImportanceSampler(nn.Module):
    def __init__(self, target, proposal, batch_shape=(1,)):
        super().__init__()
        assert callable(target) and hasattr(target, 'update')
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
        args = tuple(utils.batch_expand(arg, self.batch_shape, True)
                     if hasattr(arg, 'expand') else arg for arg in args)
        kwargs = {k: utils.batch_expand(v, self.batch_shape, True)
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

    def update(self, *args, **kwargs):
        if hasattr(self.proposal, 'update'):
            self.proposal.update(*args, **kwargs)
        return self.target.update(*args, **kwargs)

class VariationalSampler(ImportanceSampler):
    def __init__(self, target, proposal, mk_optimizer, batch_shape=(1,)):
        super().__init__(target, proposal, batch_shape)
        self._optimizer = mk_optimizer(list(self.parameters()))

    def update(self, *args):
        trace = args[-1]
        self._optimizer.step()
        self._optimizer.zero_grad()
        if len(args) > 2:
            return trace
        return ()
