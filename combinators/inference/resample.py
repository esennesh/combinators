#!/usr/bin/env python3

from discopy import cartesian, monoidal
from functools import wraps
from probtorch import RandomVariable
import torch

from .. import sampler, tracing, utils

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

def resample_box(box):
    if isinstance(box, tracing.TracedLensBox) and len(box.dom):
        box_resampler = forward_resampler(box.sample)
        return tracing.TracedLensBox(box.name, box.dom, box.cod, box_resampler,
                                     box.update, data=box.data)
    return box

def forward_resampler(f):
    @wraps(f)
    def resampler(*args, **kwargs):
        self = f.__self__
        vals, log_weight, p = f(*args, **kwargs)
        ancestors, log_weight = utils.gumbel_max_resample(log_weight)

        vals = list(cartesian.tuplify(vals))
        for i, v in enumerate(vals):
            if isinstance(v, torch.Tensor):
                vals[i] = collapsed_index_select(v, self.batch_shape, ancestors)
        vals = cartesian.untuplify(tuple(vals))

        resample = lambda rv: index_select_rv(rv, self.batch_shape, ancestors)
        p = utils.trace_map(p, resample)

        return vals, log_weight, p
    return resampler

RESAMPLING_FUNCTOR = tracing.TracedLensFunctor(lambda ob: ob, resample_box)

def resampler(diagram):
    return RESAMPLING_FUNCTOR(diagram)
