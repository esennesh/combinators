#!/usr/bin/env python3

import functools
from probtorch import Trace

from .. import lens, tracing

class TracedCopy:
    def __init__(self, n):
        self._n = n

    def __call__(self, *vals):
        return vals * self._n, (0., Trace())

class CopyFeedback:
    def __init__(self, dom, n, join=lambda x, y: x + y):
        self._dom = len(dom.upper)
        self._n = n
        self._join = join

    def __call__(self, *vals):
        assert (len(vals) - self._dom) % self._n == 0
        k = len(vals) // self._n

        chunks = [vals[self._dom + i*k:self._dom + (i+1)*k] for i
                  in range(self._n)]
        return functools.reduce(self._join, chunks)

def iid(f, n, join=lambda x, y: x + y):
    if n == 0:
        return lens.Id(f.dom)

    cod = functools.reduce(lambda x, y: x @ y, [f.dom] * n, lens.LensPRO(0))
    copy = tracing.TracedLensBox('%d-iid' % n, f.dom, cod, TracedCopy(n),
                                 CopyFeedback(f.dom, n, join))
    if f is not None:
        fs = functools.reduce(lambda g, h: g @ h, [f] * n,
                              lens.Id(lens.LensPRO(0)))
        return copy >> fs
    return copy
