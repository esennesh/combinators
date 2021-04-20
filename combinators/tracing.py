#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import dataclass
from discopy import cartesian, cat, messages, monoidal
from discopy.rigid import PRO, Ty
from functools import reduce, singledispatch
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
    def __init__(self):
        self.log_weight = 0.

    @abstractmethod
    def fold(self):
        return 0., Trace()

@dataclass
class BoxTrace(MonoidalTrace):
    retval: typing.Optional[tuple]
    log_weight: typing.Union[torch.tensor, float]
    probs: NestedTrace

    def fold(self):
        return self.log_weight, self.probs

@dataclass
class ProductTrace(MonoidalTrace):
    factors: typing.Sequence["MonoidalTrace"]

    def __post_init__(self):
        super().__init__()

    def fold(self):
        self.log_weight, probs = reduce(utils.join_tracing_states,
                                        [f.fold() for f in self.factors])
        return self.log_weight, probs

@dataclass
class CompositeTrace(MonoidalTrace):
    arrows: typing.Sequence["MonoidalTrace"]

    def __post_init__(self):
        super().__init__()

    def fold(self):
        self.log_weight, probs = reduce(utils.join_tracing_states,
                                        [a.fold() for a in self.arrows])
        return self.log_weight, probs

@dataclass
class EmptyTrace(MonoidalTrace):
    dom: lens.LensTy
    cod: lens.LensTy

    def __post_init__(self):
        super().__init__()

    def fold(self):
        return super().fold()

@monoidal.Diagram.subclass
class TracedLensDiagram(lens.LensDiagram):
    @staticmethod
    def trace(semantics, *vals, **kwargs):
        if kwargs:
            vals = vals + (kwargs,)

        result = semantics.sample(*vals)
        return result, _trace(semantics)

    @staticmethod
    def clear(semantics):
        _clear(semantics)

    def compile(self):
        return TRACED_SEMANTIC_FUNCTOR(self)

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
    pass

class TracedLensFunction(lens.LensFunction):
    def __init__(self, name, dom, cod, sample, update, **kwargs):
        self._trace = None
        super().__init__(name, dom, cod, sample, update, **kwargs)
        self.clear()

    @property
    def trace(self):
        return self._trace

    def clear(self):
        self._trace = BoxTrace(None, 0., NestedTrace())

    def sample(self, *args, **kwargs):
        self._trace = BoxTrace(*super().sample(self.trace.probs, *args,
                                               **kwargs))
        return self.trace.retval

    def update(self, *args, **kwargs):
        q, p = utils.split_latent(self.trace.probs)

        result, q = super().update(q, *args, **kwargs)
        assert all(not q[k].observed for k in q)

        self._trace.retval = None
        self._trace.probs = utils.join_traces(q, p)
        return result

class TracedBoxSemanticsFunctor(lens.BoxSemanticsFunctor):
    def box_semantics(self, box):
        if isinstance(box, TracedLensBox):
            return TracedLensFunction(box.name, box.dom, box.cod, box.sample,
                                      box.update, data=box.data)
        return super().box_semantics(box)

TRACED_SEMANTIC_FUNCTOR = TracedBoxSemanticsFunctor()

@singledispatch
def _trace(f: lens.LensSemantics):
    return EmptyTrace(f.dom, f.cod)

@_trace.register
def _(f: lens.LensProduct):
    return ProductTrace([_trace(lens) for lens in f.lenses])

@_trace.register
def _(f: lens.LensComposite):
    return CompositeTrace([_trace(lens) for lens in f.lenses])

@_trace.register
def _(f: TracedLensFunction):
    return f.trace

@singledispatch
def _clear(_: lens.LensSemantics):
    pass

@_clear.register
def _(f: lens.LensProduct):
    for l in f.lenses:
        _clear(l)

@_clear.register
def _(f: lens.LensComposite):
    for l in f.lenses:
        _clear(l)

@_clear.register
def _(f: TracedLensFunction):
    f.clear()
