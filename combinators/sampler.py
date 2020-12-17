#!/usr/bin/env python3

import collections
from contextlib import contextmanager

import probtorch
import torch
import torch.nn as nn

from .tracing import NestedTrace
from . import utils

class ImportanceSampler(nn.Module):
    def __init__(self, target, proposal, batch_shape=(1,)):
        super().__init__()
        self._batch_shape = batch_shape
        self.add_module('target', target)
        self.add_module('proposal', proposal)

    @property
    def batch_shape(self):
        return self._batch_shape

    def forward(self, *args, **kwargs):
        if args and isinstance(args[-1], dict):
            kwargs = args[-1]
            args = args[:-1]

        q = NestedTrace()
        _, q = self.proposal(q, *args, **kwargs)

        p = NestedTrace(q=q)
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

class Sampler(nn.Module):
    @property
    def name(self):
        raise NotImplementedError()

    @property
    def batch_shape(self):
        raise NotImplementedError()

    def get_model(self):
        raise NotImplementedError()

    def apply(self, f, selector):
        return self.walk(lambda m: f(m) if selector(m) else m)

    def walk(self, f):
        raise NotImplementedError()

    @contextmanager
    def score(self, ps):
        raise NotImplementedError()

    @property
    def _expander(self):
        return lambda v: utils.batch_expand(v, self.batch_shape)

    def _expand_args(self, *args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                args[i] = utils.batch_expand(arg, self.batch_shape)
            elif isinstance(arg, collections.Mapping):
                args[i] = utils.vardict_map(arg, self._expander)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = utils.batch_expand(v, self.batch_shape)
            elif isinstance(v, collections.Mapping):
                kwargs[k] = utils.vardict_map(v, self._expander)
        return tuple(args), kwargs

    def _args_vardict(self, expand=True):
        result = utils.vardict(self.state_dict(keep_vars=True))
        result = utils.vardict({k: v for k, v in result.items()
                                if '.' not in k})
        # PyTorch BUG: Parameter's don't get counted as Tensors in Normal
        for k, v in result.items():
            result[k] = v.clone()
        if expand:
            (result,), _ = self._expand_args(result)
        return result

    def args_vardict(self, expand=True):
        result = utils.vardict({})
        for module in self.children():
            if isinstance(module, Sampler):
                args = module.args_vardict(expand=expand)
                for k, v in args.items():
                    result[k] = v
        args = self._args_vardict(expand=expand)
        for k, v in args.items():
            result[k] = v
        return result

    def register_args(self, args, trainable=True):
        for k, v in utils.vardict(args).items():
            v = v.clone().detach()
            if trainable:
                self.register_parameter(k, nn.Parameter(v))
            else:
                self.register_buffer(k, v)
