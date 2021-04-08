#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass
from discopy import cartesian, cat, messages, monoidal
from discopy.rigid import PRO, Ty
from functools import reduce
import probtorch
from probtorch.stochastic import Provenance, Trace
import torch
import typing

from . import lens, utils

class NestedTrace(Trace):
    def __init__(self, q=None):
        super().__init__()
        self._q = q if q else Trace()

    def variable(self, Dist, *args, **kwargs):
        name = kwargs.get('name', None)
        if name is not None and name in self._q:
            provenance = kwargs.pop('provenance', None)
            assert provenance is not Provenance.OBSERVED
            assert isinstance(self._q[name], probtorch.RandomVariable)

            kwargs['value'] = self._q[name].value
            if self._q[name].provenance is Provenance.OBSERVED:
                kwargs['provenance'] = Provenance.OBSERVED
            else:
                kwargs['provenance'] = Provenance.REUSED

        return super().variable(Dist, *args, **kwargs)

    def conditioning_factor(self, batch_shape, nodes_next=[]):
        device = 'cpu'
        for v in self.values():
            device = v.log_prob.device
            if device != 'cpu':
                break
        dims = tuple(range(len(batch_shape)))
        null = torch.zeros(batch_shape, device=device)

        log_likelihood = self.log_joint(sample_dims=dims,
                                        nodes=list(self.conditioned()),
                                        reparameterized=False) + null
        reused = [k for k in self if self[k].provenance == Provenance.REUSED]
        log_prior = self.log_joint(sample_dims=dims, nodes=reused,
                                   reparameterized=False) -\
                    self.log_joint(sample_dims=dims, nodes=nodes_next,
                                   reparameterized=False)

        if isinstance(self._q, NestedTrace):
            log_proposal = self._q.conditioning_factor(batch_shape, reused)
        else:
            log_proposal = self._q.log_joint(sample_dims=dims, nodes=reused,
                                             reparameterized=False)
        return log_likelihood + log_prior + log_proposal

class MonoidalTrace(ABC):
    @abstractmethod
    def fold(self):
        return 0., Trace()

@dataclass
class BoxTrace(MonoidalTrace):
    retval: typing.Optional[tuple]
    log_weight: torch.tensor
    probs: NestedTrace

    def fold(self):
        return self.log_weight, self.probs

@dataclass
class ProductTrace(MonoidalTrace):
    factors: typing.Sequence["MonoidalTrace"]

    def fold(self):
        return reduce(utils.join_tracing_states,
                      [f.fold() for f in self.factors])

@dataclass
class CompositeTrace(MonoidalTrace):
    arrows: typing.Sequence["MonoidalTrace"]

    def fold(self):
        return reduce(utils.join_tracing_states,
                      [a.fold() for a in self.arrows])

def retrieve_trace(func):
    if func.data and 'tracer' in func.data:
        tracer = func.data['tracer']()
        return tracer.trace
    return TraceDiagram.UNIT(func.dom, func.cod)

TRACING_FUNCTOR = monoidal.Functor(lambda ob: ob, retrieve_trace, ob_factory=Ty,
                                   ar_factory=TraceDiagram)

def clear_tracing(func):
    if func.data and 'tracer' in func.data:
        tracer = func.data['tracer']()
        tracer.clear()
    return TraceDiagram.UNIT(func.dom, func.cod)

CLEAR_FUNCTOR = monoidal.Functor(lambda ob: ob, clear_tracing, ob_factory=Ty,
                                 ar_factory=TraceDiagram)

@dataclass
class EmptyTrace(MonoidalTrace):
    dom: lens.LensTy
    cod: lens.LensTy

    def fold(self):
        return super().fold()

@monoidal.Diagram.subclass
class TracedLensDiagram(lens.LensDiagram):
    def compile(self):
        return TRACED_SEMANTIC_FUNCTOR(self)

    @staticmethod
    def trace(semantics, *vals, **kwargs):
        if kwargs:
            vals = vals + (kwargs,)
        result = semantics.sample(*vals)
        trace = TRACING_FUNCTOR(semantics.sample)
        return result, trace

    @staticmethod
    def clear(semantics):
        CLEAR_FUNCTOR(semantics.sample)
        return semantics

    def __call__(self, *vals, **kwargs):
        return TracedLensDiagram.trace(self.compile(), *vals, **kwargs)

    @staticmethod
    def id(dom=lens.LensTy()):
        return Id(dom)

class Id(TracedLensDiagram):
    """
    Implements identity diagrams on dom inputs.
    """
    def __init__(self, dom):
        """
        >>> assert Diagram.id(42) == Id(42) == Diagram(42, 42, [], [])
        """
        assert isinstance(dom, lens.LensTy)
        super().__init__(dom, dom, [], [], layers=None)

    def __repr__(self):
        """
        >>> Id(42)
        Id(42)
        """
        return "Id({})".format(self.dom)

    def __str__(self):
        """
        >>> print(Id(42))
        Id(42)
        """
        return repr(self)

class TracedLensFunctor(monoidal.Functor):
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=lens.LensTy,
                         ar_factory=TracedLensDiagram)

class TracedLensBox(lens.LensBox, TracedLensDiagram):
    def conditioned(self, data=None):
        return TracedLensBox(self.name, self.dom, self.cod, self.sample,
                             self.update, data=data)

class TracedLensFunction(lens.LensFunction):
    def __init__(self, name, dom, cod, sample, update, **kwargs):
        self._sample_func = sample
        self._update_func = update
        self._trace = None
        traced_sample = cartesian.Box(name + '_sample', len(dom.upper),
                                      len(cod.upper), self._traced_sample,
                                      data={'tracer': lambda: self})
        traced_update = cartesian.Box(name + '_update',
                                      len(dom.upper @ cod.lower),
                                      len(dom.lower), self._traced_update,
                                      data={'tracer': lambda: self})
        super().__init__(name, dom, cod, traced_sample, traced_update, **kwargs)

    @property
    def trace(self):
        return self._trace

    def clear(self):
        self._trace = None

    def _traced_sample(self, *args, **kwargs):
        if self.data is not None:
            kwargs['data'] = self.data

        q = self.trace.box()[2] if self.trace else None
        result, log_weight, p = self._sample_func(q, *args, **kwargs)
        self._trace = TraceDiagram.BOX(result, log_weight, p, self.name)

        return result

    def _traced_update(self, *args, **kwargs):
        if self.data is not None:
            kwargs['data'] = self.data

        if self.trace:
            _, log_weight, p, _ = self.trace.box()
            q, p = utils.split_latent(p)
        else:
            q, p = None, {}
            log_weight = 0.

        result, q = self._update_func(q, *args, **kwargs)
        assert all(not q[k].observed for k in q)

        p = utils.join_traces(q, p)
        self._trace = TraceDiagram.BOX(None, log_weight, p, self.name)
        return result

    @staticmethod
    def create(box):
        if isinstance(box, TracedLensBox):
            return TracedLensFunction(box.name, box.dom, box.cod, box.sample,
                                      box.update, data=box.data)
        return lens.LensFunction.create(box)

TRACED_SEMANTIC_FUNCTOR = lens.LensFunctionFunctor(lambda ob: ob,
                                                   TracedLensFunction.create)
