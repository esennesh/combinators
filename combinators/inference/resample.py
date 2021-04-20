#!/usr/bin/env python3

from discopy import cartesian, monoidal
from functools import wraps
from probtorch import RandomVariable
import torch

from .. import lens, sampler, tracing, utils

def collapsed_index_select(tensor, batch_shape, ancestors):
    tensor, unique = utils.batch_collapse(tensor, batch_shape)
    tensor = tensor.index_select(0, ancestors)
    return tensor.reshape(batch_shape + unique)

def index_select_rv(rv, batch_shape, ancestors):
    result = rv
    if isinstance(rv, RandomVariable) and not rv.observed:
        value = collapsed_index_select(rv.value, batch_shape, ancestors)
        result = RandomVariable(rv.Dist, value, *rv.dist_args,
                                provenance=rv.provenance, mask=rv.mask,
                                **rv.dist_kwargs)
    return result

class ResamplingFunctor(lens.LensSemanticsFunctor):
    def __init__(self, root):
        self._root = root
        super().__init__(lambda ob: ob, self.arrow)

    def arrow(self, f):
        return f.fold(self.resample_box)

    def resample_box(self, box):
        if isinstance(box, tracing.TracedLensFunction):
            return lens.hook(box, post_sample=ResamplingSample(self._root))
        return box

class ResamplingSample:
    def __init__(self, root):
        self.root = root

    def __call__(this, self, vals):
        log_weight, _ = tracing._trace(this.root).fold()
        if (log_weight == 0.).all():
            return vals
        ancestors, _ = utils.gumbel_max_resample(log_weight)
        batch_shape = log_weight.shape

        vals = list(cartesian.tuplify(vals))
        for i, v in enumerate(vals):
            if isinstance(v, torch.Tensor):
                vals[i] = collapsed_index_select(v, batch_shape, ancestors)
        vals = cartesian.untuplify(tuple(vals))

        self.trace.log_weight = self.trace.log_weight.mean(dim=0, keepdim=True)
        resample = lambda rv: index_select_rv(rv, batch_shape, ancestors)
        self.trace.probs = utils.trace_map(self.trace.probs, resample)
        self.trace.retval = vals

        return vals

def resampler(semantics):
    return ResamplingFunctor(semantics)(semantics)
