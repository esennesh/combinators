#!/usr/bin/env python3

from adt import adt, Case
from discopy import cartesian, cat, messages, monoidal
from discopy.monoidal import PRO, Sum, Ty
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

@adt
class TraceDiagram:
    BOX: Case[tuple, torch.tensor, NestedTrace, str]
    PRODUCT: Case[typing.List["TraceDiagram"]]
    ARROW: Case[typing.List["TraceDiagram"]]
    UNIT: Case[Ty, Ty]
    IDENT: Case[Ty]

    def __matmul__(self, other):
        if self._key == TraceDiagram._Key.IDENT and not self.ident():
            return other
        if other._key == TraceDiagram._Key.IDENT and not other.ident():
            return self

        ls = self.product() if self._key == TraceDiagram._Key.PRODUCT else\
             [self]
        rs = other.product() if other._key == TraceDiagram._Key.PRODUCT else\
             [other]
        return TraceDiagram.PRODUCT(ls + rs)

    def __rshift__(self, other):
        if self._key == TraceDiagram._Key.IDENT:
            return other
        if other._key == TraceDiagram._Key.IDENT:
            return self

        ls = self.arrow() if self._key == TraceDiagram._Key.ARROW else [self]
        rs = other.arrow() if other._key == TraceDiagram._Key.ARROW else [other]
        return TraceDiagram.ARROW(ls + rs)

    def fold(self):
        return self.match(
            box=lambda _, log_weight, trace, __: (log_weight, trace),
            product=lambda ts: reduce(utils.join_tracing_states,
                                      [t.fold() for t in ts]),
            arrow=lambda ts: reduce(utils.join_tracing_states,
                                    [t.fold() for t in ts]),
            unit=lambda _, __: (0., {}),
            ident=lambda _: (0., {})
        )

    @staticmethod
    def id(dom):
        return TraceDiagram.IDENT(dom)

def retrieve_trace(func):
    if isinstance(func.function, TracedFunction):
        return func.function.trace
    return TraceDiagram.UNIT(func.dom, func.cod)

TRACING_FUNCTOR = monoidal.Functor(lambda ob: ob, retrieve_trace, ob_factory=Ty,
                                   ar_factory=TraceDiagram)

class TracedLensDiagram(lens.LensDiagram):
    def __call__(self, *vals, **kwargs):
        if kwargs:
            vals = vals + (kwargs,)
        morphism = TRACED_SAMPLE_FUNCTOR(self)
        result = morphism(*vals)
        trace = TRACING_FUNCTOR(morphism.sample)
        return result, trace.fold()

    @staticmethod
    def upgrade(old):
        return TracedLensDiagram(old.dom, old.cod, old.boxes, old.offsets,
                                 old.layers)

class TracedLensBox(lens.LensBox, TracedLensDiagram):
    pass

class TracedFunction:
    def __init__(self, name, function):
        self._name = name
        self._function = function
        self._trace = None

    @property
    def trace(self):
        trace = self._trace
        self._trace = None
        return trace

    def __call__(self, *vals):
        results, (log_weight, trace) = self._function(*vals)
        self._trace = TraceDiagram.BOX(results, log_weight, trace, self._name)
        return results

class TracedLensFunction(lens.LensFunction):
    def __init__(self, name, dom, cod, sample, update):
        sample = cartesian.Box(sample.name, sample.dom, sample.cod,
                               TracedFunction(sample.name, sample.function))
        super().__init__(name, dom, cod, sample, update)

    @staticmethod
    def create(box):
        base = lens.LensFunction.create(box)
        if isinstance(box, TracedLensBox):
            return TracedLensFunction(box.name, box.dom, box.cod, base.sample,
                                      base.update)
        return base

TRACED_SAMPLE_FUNCTOR = lens.LensFunctor(lambda ob: ob,
                                         TracedLensFunction.create)
