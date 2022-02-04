#!/usr/bin/env python3

from functools import reduce
import inspect
import probtorch
import torch
from discopy import cartesian, messages, monoidal, wiring

from . import lens, utils

class WeightedSampler(torch.nn.Module):
    def __init__(self, target, batch_shape=(1,)):
        super().__init__()
        sig = inspect.signature(target.forward)

        self._batch_shape = batch_shape
        self._pass_data = 'data' in sig.parameters
        self._pass_batch_shape = 'batch_shape' in sig.parameters

        self.add_module('target', target)

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def pass_data(self):
        return self._pass_data

    def expand_args(self, *args, **kwargs):
        args = tuple(utils.batch_expand(arg, self.batch_shape, True)
                     if hasattr(arg, 'expand') else arg for arg in args)
        kwargs = {k: utils.batch_expand(v, self.batch_shape, True)
                     if hasattr(v, 'expand') else v for k, v in kwargs.items()}
        return args, kwargs

    def forward(self, q, *args, **kwargs):
        if args and isinstance(args[-1], dict):
            kwargs = {**args[-1], **kwargs}
            args = args[:-1]
        if self._pass_batch_shape:
            kwargs['batch_shape'] = self.batch_shape
        if not self._pass_data and 'data' in kwargs:
            del kwargs['data']

        args, kwargs = self.expand_args(*args, **kwargs)

        p = probtorch.NestedTrace(q=q)
        result = self.target(p, *args, **kwargs)

        device = 'cpu'
        for v in p.values():
            device = v.log_prob.device
            if device != 'cpu':
                break
        dims = tuple(range(len(self.batch_shape)))
        null = torch.zeros(self.batch_shape, device=device)

        log_weight = null + p.log_proper_weight(sample_dims=dims)
        assert log_weight.shape == self.batch_shape

        return cartesian.tuplify(result), p, log_weight

class ImportanceSemanticsFunctor(lens.CartesianSemanticsFunctor):
    @classmethod
    def semantics(cls, f):
        if isinstance(f, ImportanceBox):
            return ImportanceWiringBox(f.name, f.dom, f.cod, f.target,
                                       f.proposal, data=f.data)
        return super(ImportanceSemanticsFunctor, cls).semantics(f)

class ImportanceBox(lens.Box):
    IMPORTANCE_SEMANTICS = ImportanceSemanticsFunctor()
    def __init__(self, name, dom, cod, target, proposal, data={}):
        assert isinstance(target, torch.nn.Module)
        self._target = target
        assert isinstance(proposal, torch.nn.Module)
        self._proposal = proposal

        super().__init__(name, dom, cod, data)

    @property
    def target(self):
        return self._target

    @property
    def proposal(self):
        return self._proposal

class ImportanceWiringBox(lens.CartesianWiringBox):
    def __init__(self, name, dom, cod, target, proposal, data={}):
        assert isinstance(target, torch.nn.Module)
        self._target = target
        assert isinstance(proposal, torch.nn.Module)
        self._proposal = proposal
        self._cache = utils.TensorialCache(None, self._target.forward)

        super().__init__(name, dom, cod, self.filter, self.smooth, data=data)

    def peek(self):
        return self._cache.peek()

    def filter(self, *args, **kwargs):
        if self._target.pass_data:
            _, data = self._target.expand_args((), **self.data)
            kwargs = {**data, **kwargs}

        result, _, _ = self._cache(None, *args, **kwargs)
        return result

    # TODO: incorporate replay
    def smooth(self, *args, **kwargs):
        if self._target.pass_data:
            _, data = self._target.expand_args((), **self.data)
            kwargs = {**data, **kwargs}
        dims = tuple(range(len(self._target.batch_shape)))
        fwd = args[:len(self.dom.upper)]
        cached_fwd = (None, *fwd)

        # Retrieve the stored target trace from the cache, initializing by
        # filtering if necessary.
        _, p, log_v = self._cache(*cached_fwd, **kwargs)
        log_orig = p.log_joint(sample_dims=dims)

        # Rescore the original trace under the proposal kernel
        q = probtorch.NestedTrace(q=p)
        self._proposal.forward(q, *args, **kwargs)
        log_rk = q.log_joint(sample_dims=dims)

        # Draw the new trace and the feedback from the proposal kernel
        q = probtorch.Trace()
        feedback = self._proposal.forward(q, *args, **kwargs)
        log_fk = q.log_joint(sample_dims=dims)

        # Score the new trace under the target program
        result, p, _ = self._target.forward(q, *fwd, **kwargs)
        log_update = p.log_joint(sample_dims=dims)

        log_v = log_v + (log_update + log_rk) - (log_orig + log_fk)
        self._cache[(cached_fwd, kwargs)] = (result, p, log_v)

        return feedback

def importance_box(name, target, batch_shape, proposal, dom, cod, data={}):
    assert not isinstance(dom, lens.Ty) and not isinstance(cod, lens.Ty)
    dom = dom & monoidal.PRO(len(dom))
    cod = cod & monoidal.PRO(len(cod))
    target = WeightedSampler(target, batch_shape)

    return ImportanceBox(name, dom, cod, target, proposal, data=data)

def compile(diagram):
    return ImportanceBox.IMPORTANCE_SEMANTICS(diagram)

def filter(diagram, *args, **kwargs):
    if not isinstance(diagram, wiring.Diagram):
        diagram = compile(diagram)
    return lens.getter(diagram)(*args, **kwargs)

def smooth(diagram, *args, **kwargs):
    if not isinstance(diagram, wiring.Diagram):
        diagram = compile(diagram)
    return lens.putter(diagram)(*args, **kwargs)

def __trace_falgebra__(f):
    if isinstance(f, ImportanceWiringBox):
        _, (_, p, log_weight) = f.peek()
        return p, log_weight
    if isinstance(f, (wiring.Id, lens.CartesianWiringBox)):
        return probtorch.Trace(), 0.
    if isinstance(f, wiring.Sequential):
        return reduce(utils.join_tracing_states, f.arrows)
    if isinstance(f, wiring.Parallel):
        return reduce(utils.join_tracing_states, f.factors)
    raise TypeError(messages.type_err(wiring.Diagram, f))

def trace(diagram):
    assert isinstance(diagram, wiring.Diagram)
    return diagram.collapse(__trace_falgebra__)
