#!/usr/bin/env python3

from adt import adt, Case
from discopy import cartesian, cat, messages, monoidal
from discopy.monoidal import PRO, Sum, Ty
import probtorch
from probtorch.stochastic import Provenance, Trace
import torch

from . import lens, utils

@adt
class TraceDiagram:
    BOX: Case[tuple, torch.tensor, Trace, str]
    PRODUCT: Case["TraceDiagram", "TraceDiagram"]
    ARROW: Case["TraceDiagram", "TraceDiagram"]
    UNIT: Case[Ty, Ty]

    def __matmul__(self, other):
        return TraceDiagram.PRODUCT(self, other)

    def __rshift__(self, other):
        return TraceDiagram.ARROW(self, other)

    def fold(self):
        return self.match(
            box=lambda _, log_weight, trace, __: (log_weight, trace),
            product=lambda tx, ty: tx.join(ty),
            arrow=lambda tx, ty: tx.join(ty),
            unit=lambda _, __: (0., {})
        )

    def join(self, other):
        log_weightx, tracex = self.fold()
        log_weighty, tracey = other.fold()
        return (log_weightx + log_weighty, utils.join_traces(tracex, tracey))

    @staticmethod
    def id(dom):
        return TraceDiagram.UNIT(dom, dom)

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
    def __init__(self, name, dom, cod, sample, update):
        super().__init__(name, dom, cod, sample, update)


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
