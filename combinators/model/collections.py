#!/usr/bin/env python3

import functools
from probtorch import Trace

from .. import lens, tracing

def iid(dom, n, f=None):
    if not isinstance(dom, lens.LensTy):
        dom = lens.LensPRO(dom)
    if f is not None:
        assert dom == f.dom
    if n == 0:
        return lens.Id(dom)

    def traced_copy(*vals):
        return vals * n, (0., Trace())
    def copy_feedback(*vals):
        return vals[len(dom):]
    cod = functools.reduce(lambda x, y: x @ y, [dom] * n, lens.LensPRO(0))
    copy = tracing.TracedLensBox('%d-iid' % n, dom, cod, traced_copy,
                                 copy_feedback)
    if f is not None:
        fs = functools.reduce(lambda g, h: g @ h, [f] * n,
                              lens.Id(lens.LensPRO(0)))
        return copy >> fs
    return copy
