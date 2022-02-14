#!/usr/bin/env python3

import functools

from .. import lens

def iid(f, n):
    if n == 0:
        return lens.Id(f.dom)

    copy = lens.Copy(f.dom)
    fs = functools.reduce(lambda g, h: g @ h, [f] * n,
                          lens.Id(lens.PRO(0)))
    return copy >> fs

def sequential(f, n):
    return functools.reduce(lambda x, y: x >> y, [f] * n, lens.Id(f.dom))

def parameterized_ssm(params, state, f):
    assert f.dom == state @ params

    return lens.Id(state) @ lens.Copy(params) >> (f @ lens.Id(params))
