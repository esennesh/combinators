#!/usr/bin/env python3

import collections
from functools import reduce

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import probtorch
from probtorch.util import log_mean_exp, log_sum_exp
import torch
from torch.distributions import Gumbel
import torch.nn as nn
from torch.nn.functional import log_softmax

EMPTY_TRACE = collections.defaultdict(lambda: None)

class TensorialCache:
    def __init__(self, size, func):
        self._cache = collections.deque(maxlen=size)
        self._func = func

    def __call__(self, *args, **kwargs):
        val = self.get((args, kwargs))
        if val is not None:
            return val

        val = self._func(*args, **kwargs)
        self[(args, kwargs)] = val
        return val

    def __getitem__(self, key):
        args, kwargs = key
        for (c_args, c_kwargs), val in self._cache:
            if all(tensorial_eq(c, a) for (c, a) in zip(c_args, args)) and\
               all(tensorial_eq(c_kwargs[k], kwargs[k]) for k in kwargs):
                return val
        raise KeyError(key)

    def __setitem__(self, key, val):
        args, kwargs = key
        for i, ((c_args, c_kwargs), _) in enumerate(self._cache):
            if all(tensorial_eq(c, a) for (c, a) in zip(c_args, args)) and\
               all(tensorial_eq(c_kwargs[k], kwargs[k]) for k in kwargs):
                self._cache[i] = (key, val)
                return
        self._cache.append((key, val))

    def __contains__(self, key):
        return self.get(key) is not None

    def get(self, key):
        try:
            return self[key]
        except KeyError:
            return None

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def __bool__(self):
        return len(self) > 0

    def peek(self):
        return self._cache[-1]

def tensorial_eqs(xs, ys):
    return all(tensorial_eq(x, y) for (x, y) in zip(xs, ys))

def tensorial_dict_eq(xs, ys):
    return all(tensorial_eq(xs[k], ys[k]) for k in ys)

def tensorial_eq(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return (x == y).all()
    if isinstance(x, probtorch.Trace) and isinstance(y, probtorch.Trace):
        if set(x.variables()) != set(y.variables()):
            return False
        return all(tensorial_eq(x[v].value, y[v].value) for v in x.variables())
    return x == y

def batch_where(condition, yes, no, batch_shape):
    yes, unique = batch_collapse(yes, batch_shape)
    no, _ = batch_collapse(no, batch_shape)
    ite = particle_index(torch.stack((no, yes), dim=1), condition)
    return ite.reshape(batch_shape + unique)

def is_number(tensor):
    return isnum(tensor).all()

def isnum(tensor):
    return ~(torch.isnan(tensor) | torch.isinf(tensor))

def reused_variable(px, py, k):
    reused_px = isinstance(px[k], probtorch.RandomVariable) and\
                px[k].provenance == probtorch.stochastic.Provenance.REUSED
    reused_py = isinstance(py[k], probtorch.RandomVariable) and\
                py[k].provenance == probtorch.stochastic.Provenance.REUSED
    return reused_px or reused_py

def gumbel_max_resample(log_weights):
    particle_logs, _ = batch_collapse(log_weights, log_weights.shape)
    ancestors = gumbel_max_categorical(particle_logs, particle_logs.shape)
    log_marginal = batch_expand(log_mean_exp(particle_logs, dim=0),
                                log_weights.shape)
    return ancestors, log_marginal

def gumbel_max_categorical(log_probs, sample_shape):
    k = log_probs.shape[0]
    dist = Gumbel(torch.zeros(k, device=log_probs.device),
                  torch.ones(k, device=log_probs.device))
    gumbels = dist.sample(sample_shape)
    return torch.argmax(gumbels + log_probs, dim=-1)

def normalize_weights(log_weights):
    batch_shape = log_weights.shape
    log_weights, _ = batch_collapse(log_weights, batch_shape)
    log_weights = log_softmax(log_weights, dim=0)
    return log_weights.reshape(*batch_shape)

def unique_shape(tensor, shape):
    for i, dim in enumerate(tensor.shape):
        if i >= len(shape) or (dim != 1 and shape[i] != dim):
            return tensor.shape[i:]
    return ()

def batch_log_sum_exp(tensor):
    batch_tensor, _ = batch_collapse(tensor, tensor.shape)
    return log_sum_exp(batch_tensor, dim=0)

def batch_sum(tensor):
    batch_tensor, _ = batch_collapse(tensor, tensor.shape)
    return batch_tensor.sum(dim=0)

def batch_mean(tensor, batch_shape=None):
    if not batch_shape:
        batch_shape = tensor.shape
    batch_tensor, _ = batch_collapse(tensor, batch_shape)
    return batch_tensor.mean(dim=0)

def batch_marginalize(tensor):
    batch_tensor, _ = batch_collapse(tensor, tensor.shape)
    return log_mean_exp(batch_tensor, dim=0)

def batch_collapse(tensor, shape):
    collapsed = reduce(lambda x, y: x * y, shape)
    unique = unique_shape(tensor, shape)
    return tensor.reshape((collapsed,) + unique), unique

def particle_matmul(matrices, vectors):
    return torch.bmm(matrices, vectors.unsqueeze(-1)).squeeze(-1)

def slice_trace(trace, key, forwards=True):
    result = probtorch.Trace()
    items = trace.items() if forwards else reversed(trace.items())
    for k, v in items:
        if k == key:
            break
        result[k] = v
    return result

def trace_map(trace, f):
    p = probtorch.Trace()
    for k, v in trace.items():
        p[k] = f(trace[k])
    return p

def trace_filter(trace, f):
    p = probtorch.Trace()
    for k, v in trace.items():
        if f(k, v):
            p[k] = v
    return p

def split_latent(trace):
    latent = trace_filter(trace, lambda k, v: not v.observed)
    observed = trace_filter(trace, lambda k, v: v.observed)
    return latent, observed

class TracingMerger:
    def __init__(self):
        self._log_weight = 0.
        self._p = probtorch.Trace()
        self._names = collections.defaultdict(int)

    @property
    def p(self):
        return self._p

    @property
    def log_weight(self):
        return self._log_weight

    def __call__(self, p, log_weight):
        for k, v in p.items():
            stem, _, _ = k.partition('_')
            n = self._names[stem]
            if n:
                k = stem + '_' + str(n)
            self._p[k] = v
            self._names[stem] += 1
        self._log_weight = self._log_weight + log_weight

def marginalize_all(log_prob):
    for _ in range(len(log_prob.shape)):
        log_prob = log_mean_exp(log_prob, dim=0)
    return log_prob

def try_rsample(dist):
    if dist.has_rsample:
        return dist.rsample()
    return dist.sample()

def broadcastable_sizes(a, b):
    result = ()
    for (dima, dimb) in reversed(list(zip(a, b))):
        if dima == dimb or dimb == 1:
            result += (dima,)
        elif dima == 1:
            result += (dimb,)
        else:
            break
    return result

def broadcastable_shape(a, b):
    return broadcastable_sizes(a.shape, b.shape)

def conjunct_event_shape(tensor, batch_dims):
    while len(tensor.shape) > batch_dims:
        tensor = tensor.sum(dim=batch_dims)
    return tensor

def conjunct_events(conjunct_log_prob, log_prob):
    batch_dims = len(broadcastable_shape(conjunct_log_prob, log_prob))
    return conjunct_log_prob + conjunct_event_shape(log_prob, batch_dims)

def dict_lookup(d):
    return lambda name, dist: d.get(name, None)

def plot_evidence_bounds(bounds, lower=True, figsize=(10, 10), scale='linear'):
    epochs = range(len(bounds))
    bound_name = 'ELBO' if lower else 'EUBO'

    free_energy_fig = plt.figure(figsize=figsize)

    plt.plot(epochs, bounds, 'b-', label='Data')
    plt.legend()

    free_energy_fig.tight_layout()
    plt.title('%s over training' % bound_name)
    free_energy_fig.axes[0].set_xlabel('Epoch')
    free_energy_fig.axes[0].set_ylabel('%s (nats)' % bound_name)
    free_energy_fig.axes[0].set_yscale(scale)

    plt.show()

def batch_expand(tensor, shape, check=False):
    if not shape or (check and not unique_shape(tensor, shape)):
        return tensor
    return tensor.expand(*shape, *unique_shape(tensor, shape))

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

def vardict_keys(vdict):
    first_level = [k.rsplit('__', 1)[0] for k in vdict.keys()]
    return list(set(first_level))

def walk_trie(trie, keys=[]):
    while len(keys) > 0:
        trie = trie[keys[0]]
        keys = keys[1:]
    return trie

PARAM_TRANSFORMS = {
    'concentration': lambda v: ('concentration', nn.functional.softplus(v)),
    'precision': lambda v: ('scale', nn.functional.softplus(v)**(-1.)),
    'scale': lambda v: ('scale', nn.functional.softplus(v)),
}

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip
