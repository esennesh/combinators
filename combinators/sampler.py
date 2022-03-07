#!/usr/bin/env python3

from functools import lru_cache, partial, reduce
import inspect
import probtorch
import torch
from discopy import cartesian, messages, monoidal, wiring

from . import lens, signal, utils

class WeightedSampler(torch.nn.Module):
    def __init__(self, target, proposal, batch_shape=(1,), particle_shape=(1,)):
        super().__init__()
        sig = inspect.signature(target.forward)
        self._batch_shape = batch_shape
        self._particle_shape = particle_shape
        self._pass_data = 'data' in sig.parameters
        self._pass_batch_shape = 'batch_shape' in sig.parameters
        self._pass_particle_shape = 'particle_shape' in sig.parameters

        self._cache = utils.TensorialCache(None, self.forward)

        self.add_module('proposal', proposal)
        self.add_module('target', target)

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def particle_shape(self):
        return self._particle_shape

    @property
    def pass_data(self):
        return self._pass_data

    @property
    def cache(self):
        return self._cache

    def peek(self):
        return self._cache.peek()

    def clear(self):
        return self._cache.clear()

    @property
    def inference_state(self):
        if self.cache:
            _, (_, p, log_weight) = self.peek()
            return p, log_weight
        return None

    def expand_args(self, *args, **kwargs):
        args = tuple(utils.particle_expand(arg, self.particle_shape, True)
                     if hasattr(arg, 'expand') else arg for arg in args)
        kwargs = {k: utils.particle_expand(v, self.particle_shape, True)
                     if hasattr(v, 'expand') else v for k, v in kwargs.items()}
        return args, kwargs

    def forward(self, q, *args, **kwargs):
        if args and isinstance(args[-1], dict):
            kwargs = {**args[-1], **kwargs}
            args = args[:-1]
        if self._pass_batch_shape:
            kwargs['batch_shape'] = self.batch_shape
        if self._pass_particle_shape:
            kwargs['particle_shape'] = self.particle_shape
        if not self._pass_data and 'data' in kwargs:
            del kwargs['data']

        args, kwargs = self.expand_args(*args, **kwargs)

        p = probtorch.NestedTrace(q=q)
        result = self.target(p, *args, **kwargs)

        dims = tuple(range(len(self.particle_shape)))
        log_weight = p.log_proper_weight(sample_dims=dims, batch_dim=len(dims))
        if torch.is_tensor(log_weight):
            assert log_weight.shape == (self.particle_shape + self.batch_shape)

        return cartesian.tuplify(result), p, log_weight

    def fill_args(self, *args, **kwargs):
        cached, _ = self.peek()
        args = tuple(stored if actual is None else actual for (stored, actual)
                     in zip(cached[0][1:], args))
        kwargs = {
            k: cached[1][k] if k not in kwargs or kwargs[k] is None else
               kwargs[k] for k in kwargs.keys() | cached[1].keys()
        }
        return args, kwargs

    def filter(self, *args, **kwargs):
        return self._cache(None, *args, **kwargs)[0]

    def replay(self, kont, *args, **kwargs):
        args, kwargs = self.fill_args(*args, **kwargs)

        if ((None, *args), kwargs) in self._cache:
            result, p, _ = self._cache(None, *args, **kwargs)
            kontinue = False
        else:
            _, (stored_result, q, _) = self.peek()
            result, p, log_weight = self.forward(q, *args, **kwargs)
            self._cache[((None, *args), kwargs)] = (result, p, log_weight)
            kontinue = not utils.tensorial_eqs(stored_result, result)

        if kont and kontinue:
            for wire, r in zip(kont, result):
                wire.update(r)

    def feedback(self):
        (args, kwargs), (_, p, _) = self.cache.peek()
        return self.proposal.feedback(p, *args[1:], **kwargs)

    def smooth(self, args, wires, **kwargs):
        dims = tuple(range(len(self.particle_shape)))

        # Retrieve the stored target trace from the cache, initializing by
        # filtering if necessary.
        _, p, log_v = self.cache(None, *args, **kwargs)
        log_orig = p.log_joint(sample_dims=dims, batch_dim=len(dims))

        # Retrieve the feedback corresponding to the stored target trace
        feedback = ()
        for wire in wires:
            feedback = feedback + wire()

        # Rescore the original trace under the proposal kernel
        q = probtorch.NestedTrace(q=p)
        self.proposal(q, *args, *feedback, **kwargs)
        log_rk = q.log_joint(sample_dims=dims, batch_dim=len(dims))

        # Draw the new trace and the feedback from the proposal kernel
        q = probtorch.Trace()
        self.proposal(q, *args, *feedback, **kwargs)
        log_fk = q.log_joint(sample_dims=dims, batch_dim=len(dims))

        # Score the new trace under the target program
        result, p, _ = self(q, *args, **kwargs)
        log_update = p.log_joint(sample_dims=dims, batch_dim=len(dims))

        log_v = log_v + (log_update + log_rk) - (log_orig + log_fk)
        self.cache[((None, *args), kwargs)] = (result, p, log_v)

        # Update the downstream computation
        for wire, r in zip(wires, result):
            wire.update(r)

        # Return the feedback corresponding to the new target trace
        return self.feedback, partial(self.replay, wires)

class ImportanceWiringSemantics(lens.CartesianWiringSemantics):
    @classmethod
    def semantics(cls, f):
        if isinstance(f, ImportanceBox):
            return ImportanceWiringBox(f.name, f.dom, f.cod, f.sampler,
                                       data=f.data)
        return super(ImportanceWiringSemantics, cls).semantics(f)

class ImportanceBox(lens.Box):
    IMPORTANCE_SEMANTICS = ImportanceWiringSemantics()
    def __init__(self, name, dom, cod, sampler, data={}):
        assert isinstance(sampler, WeightedSampler)
        self._sampler = sampler

        super().__init__(name, dom, cod, data)

    @property
    def sampler(self):
        return self._sampler

class Copy(lens.Copy):
    def __init__(self, dom, n=2):
        super().__init__(dom, n=n, join=self.join)

    @staticmethod
    def join(sx, sy):
        def sig(*arg):
            x = cartesian.untuplify(*sx(*arg))
            y = cartesian.untuplify(*sy(*arg))
            if torch.is_tensor(x) and torch.is_tensor(y):
                if len(x.shape) == len(y.shape):
                    return torch.stack((x, y), dim=2)
                if len(x.shape) < len(y.shape):
                    return torch.cat((x.unsqueeze(2), y), dim=2)
                if len(y.shape) < len(x.shape):
                    return torch.cat((x, y.unsqueeze(2)), dim=2)
            return (x, y)
        def sig_update(*arg):
            sx.update(*arg)
            sy.update(*arg)
        return signal.Signal(sx.dom, sig, sig_update)

class ImportanceWiringBox(lens.CartesianWiringBox):
    def __init__(self, name, dom, cod, sampler, data={}):
        assert isinstance(sampler, WeightedSampler)
        self._sampler = sampler

        super().__init__(name, dom, cod, self.filter, self.smooth, data=data)

    @property
    def sampler(self):
        return self._sampler

    def filter(self, *args, **kwargs):
        if self.sampler.pass_data:
            kwargs = {**self.data, **kwargs}

        return self.sampler.filter(*args, **kwargs)

    def smooth(self, *args, **kwargs):
        if self.sampler.pass_data:
            _, data = self.sampler.expand_args((), **self.data)
            kwargs = {**data, **kwargs}

        fwd = args[:len(self.dom.upper)]
        if len(fwd) != len(self.dom.upper):
            raise TypeError(messages.expected_input_length(self, fwd))

        wires = args[len(self.dom.upper):]
        if len(wires) != len(self.cod.lower):
            raise TypeError(messages.expected_input_length(self, wires))

        feedback, replay = self.sampler.smooth(fwd, wires, **kwargs)
        return signal.Signal(len(self.dom.lower), feedback, replay).split()

def importance_box(name, target, proposal, batch_shape, particle_shape, dom,
                   cod, data={}):
    if not isinstance(dom, lens.Ty):
        dom = dom & monoidal.PRO(len(dom))
    assert len(dom.upper) == len(dom.lower)
    if not isinstance(cod, lens.Ty):
        cod = cod & monoidal.PRO(len(cod))
    assert len(cod.upper) == len(cod.lower)

    sampler = WeightedSampler(target, proposal, batch_shape, particle_shape)
    return ImportanceBox(name, dom, cod, sampler, data=data)

@lru_cache(maxsize=None)
def compile(diagram):
    return ImportanceBox.IMPORTANCE_SEMANTICS(diagram)

def filtering(diagram, precompile=True):
    if not isinstance(diagram, wiring.Diagram):
        diagram = compile(diagram)
    return lens.getter(diagram, precompile)

def filter(diagram, *args, **kwargs):
    return filtering(diagram)(*args, **kwargs)

def smoothing(diagram, precompile=True):
    if not isinstance(diagram, wiring.Diagram):
        diagram = compile(diagram)
    return lens.putter(diagram, precompile)

def smooth(diagram, *args, **kwargs):
    return smoothing(diagram)(*args, **kwargs)

def trace(diagram):
    assert isinstance(diagram, wiring.Diagram)
    merge = utils.TracingMerger()
    for f in diagram:
        if isinstance(f, ImportanceWiringBox):
            inference = f.sampler.inference_state
            if inference:
                merge(*inference)
    return merge.p, merge.log_weight

def clear(diagram):
    assert isinstance(diagram, wiring.Diagram)
    for f in diagram:
        if isinstance(f, ImportanceWiringBox):
            f.sampler.clear()

def __params_falgebra__(f):
    if isinstance(f, ImportanceWiringBox):
        target, proposal = f.sampler.target, f.sampler.proposal
        return set(target.parameters()), set(proposal.parameters())
    if isinstance(f, (wiring.Id, lens.CartesianWiringBox)):
        return set(), set()
    if isinstance(f, wiring.Sequential):
        return reduce(lambda x, y: (x[0] | y[0], x[1] | y[1]), f.arrows)
    if isinstance(f, wiring.Parallel):
        return reduce(lambda x, y: (x[0] | y[0], x[1] | y[1]), f.factors)
    raise TypeError(messages.type_err(wiring.Diagram, f))

def parameters(diagram):
    assert isinstance(diagram, wiring.Diagram)
    return diagram.collapse(__params_falgebra__)
