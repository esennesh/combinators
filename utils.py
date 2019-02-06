#!/usr/bin/env python3

import collections
import flatdict

import matplotlib.pyplot as plt
import probtorch
from probtorch.util import log_mean_exp
import torch
import torch.nn as nn

EMPTY_TRACE = collections.defaultdict(lambda: None)

def trace_map(trace, f):
    p = probtorch.Trace()
    for k, v in trace.items():
        p[k] = f(trace[k])
    return p

def join_traces(first, second):
    p = probtorch.Trace()
    for k, v in first.items():
        p[k] = v
    for k, v in second.items():
        p[k] = v
    return p

def marginalize_all(log_prob):
    for _ in range(len(log_prob.shape)):
        log_prob = log_mean_exp(log_prob, dim=0)
    return log_prob

def try_rsample(dist):
    if dist.has_rsample:
        return dist.rsample()
    return dist.sample()

def shared_shape(a, b):
    result = ()
    for (dim, dimb) in zip(a.shape, b.shape):
        if dim == dimb or dim == 1 or dimb == 1:
            result += (dim,)
        else:
            break
    return result

def conjunct_event_shape(tensor, batch_dims):
    while len(tensor.shape) > batch_dims:
        tensor = tensor.sum(dim=batch_dims)
    return tensor

def conjunct_events(conjunct_log_prob, log_prob):
    batch_dims = len(shared_shape(conjunct_log_prob, log_prob))
    return conjunct_log_prob + conjunct_event_shape(log_prob, batch_dims)

def dict_lookup(d):
    return lambda name, dist: d.get(name, None)

def plot_evidence_bounds(bounds, lower=True, figsize=(10, 10)):
    epochs = range(len(bounds))
    bound_name = 'ELBO' if lower else 'EUBO'

    free_energy_fig = plt.figure(figsize=figsize)

    plt.plot(epochs, bounds, 'b-', label='Data')
    plt.legend()

    free_energy_fig.tight_layout()
    plt.title('%s over training' % bound_name)
    free_energy_fig.axes[0].set_xlabel('Epoch')
    free_energy_fig.axes[0].set_ylabel('%s (nats)' % bound_name)

    plt.show()

def batch_expand(tensor, shape):
    if not shape:
        return tensor
    tensor = tensor.expand(shape[-1], *tensor.shape)
    if len(shape) > 1:
        return batch_expand(tensor, shape[:-1])
    return tensor

def vardict_map(vdict, func):
    result = vardict()
    for k, v in vdict.items():
        result[k] = func(v)
    return result

def vardict_particle_index(vdict, indices):
    return vardict_map(vdict, lambda v: particle_index(v, indices))

def vardict_index_select(vdict, indices, dim=0):
    return vardict_map(vdict, lambda v: v.index_select(dim, indices))

def counterfactual_log_joint(p, q, rvs):
    return sum([p[rv].dist.log_prob(q[rv].value.to(p[rv].value)) for rv in rvs
                if rv in p])

def optional_to(tensor, other):
    if isinstance(tensor, probtorch.stochastic.RandomVariable):
        return tensor.value.to(other)
    elif tensor is not None:
        return tensor.to(other)
    return tensor

def particle_index(tensor, indices):
    indexed_tensors = [t[indices[particle]] for particle, t in
                       enumerate(torch.unbind(tensor, 0))]
    return torch.stack(indexed_tensors, dim=0)

def relaxed_categorical(probs, name, this=None):
    if this.training:
        return this.trace.relaxed_one_hot_categorical(0.66, probs=probs,
                                                      name=name)
    return this.trace.variable(torch.distributions.Categorical, probs,
                               name=name)

def weighted_sum(tensor, indices):
    if len(tensor.shape) == 2:
        return indices @ tensor
    weighted_dim = len(indices.shape) - 1
    while len(indices.shape) < len(tensor.shape):
        indices = indices.unsqueeze(-1)
    return (indices * tensor).sum(dim=weighted_dim)

def relaxed_index_select(tensor, probs, name, dim=0, this=None):
    indices = relaxed_categorical(probs, name, this=this)
    if this.training:
        return weighted_sum(tensor, indices), indices
    return tensor.index_select(dim, indices), indices

def relaxed_vardict_index_select(vdict, probs, name, dim=0, this=None):
    indices = relaxed_categorical(probs, name, this=this)
    result = vardict()
    for k, v in vdict.items():
        result[k] = weighted_sum(v, indices) if this.training\
                    else v.index_select(dim, indices)
    return result, indices

def relaxed_particle_index(tensor, indices, this=None):
    if this.training:
        indexed_tensors = [indices[particle] @ t for particle, t in
                           enumerate(torch.unbind(tensor, 0))]
        return torch.stack(indexed_tensors, dim=0)
    return particle_index(tensor, indices)

def map_tensors(f, *args):
    for arg in args:
        if isinstance(arg, torch.Tensor):
            yield f(arg)
        else:
            yield arg

def vardict(existing=None, to=()):
    vdict = flatdict.FlatDict(delimiter='__')
    if existing:
        for k, v in existing.items():
            vdict[k] = batch_expand(v, to) if to else v
    return vdict

def vardict_keys(vdict):
    first_level = [k.rsplit('__', 1)[0] for k in vdict.keys()]
    return list(set(first_level))

def walk_trie(trie, keys=[]):
    while len(keys) > 0:
        trie = trie[keys[0]]
        keys = keys[1:]
    return trie

PARAM_TRANSFORMS = {
    'scale': nn.functional.softplus,
    'concentration': nn.functional.softplus,
}

def _parameterize_trace_methods(transforms=PARAM_TRANSFORMS):
    import inspect as _inspect

    for k, v in _inspect.getmembers(probtorch.Trace):
        if _inspect.isfunction(v):
            args = _inspect.signature(v).parameters.keys()
            if 'name' in args and 'value' in args:
                def param_sample(self, params, name=None, value=None, k=k,
                                 **kwargs):
                    params = {**params[name].copy(), **kwargs}
                    for arg, val in params.items():
                        matches = [k for k in transforms if k in arg]
                        if matches:
                            params[arg] = transforms[matches[0]](val)
                    return getattr(self, k)(name=name, value=value, **params)
                setattr(probtorch.Trace, 'param_' + k, param_sample)

_parameterize_trace_methods()
