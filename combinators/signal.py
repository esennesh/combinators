#!/usr/bin/env python3

from abc import abstractmethod
from functools import reduce, wraps
import itertools

from discopy import cartesian, messages, monoidal, rigid

class Signal(rigid.Box):
    """
    Wraps Python functions that can be sliced into separate wires, each of which
    may have a cached default value, with domain and codomain information.
    """
    def __init__(self, dom, function, update):
        assert callable(function)
        self._function = function
        assert callable(update)
        self._update = update

        super().__init__(repr(function), monoidal.PRO(dom), monoidal.PRO(dom))

    def __repr__(self):
        return "Signal(dom={}, function={}, update={})".format(
            self.dom, repr(self._function), repr(self._update))

    def __str__(self):
        return repr(self)

    @property
    def update(self):
        return self._update

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    @staticmethod
    def id(dom=0):
        return Signal(dom, cartesian.untuplify, lambda *xs: None)

    def tensor(self, *others):
        assert len(others) == 1
        other = others[0]
        if not isinstance(other, Signal):
            raise TypeError(messages.type_err(Signal, other))
        dom = self.dom @ other.dom
        if dom == self.dom:
            return self
        if dom == other.dom:
            return other

        def product(*vals):
            vals0 = cartesian.tuplify(self(*vals[:len(self.dom)]))
            vals1 = cartesian.tuplify(other(*vals[len(self.dom):]))
            return cartesian.untuplify(*(vals0 + vals1))
        def product_eff(*vals):
            self.update(*vals[:len(self.dom)])
            other.update(*vals[len(self.dom):])
        return Signal(dom, product, product_eff)

    def __getitem__(self, key):
        if isinstance(key, int):
            def index_signal(val):
                args = [None for _ in range(len(self.dom))]
                args[key] = val
                return self(*args)[key]
            def index_update(val):
                args = [None for _ in range(len(self.dom))]
                args[key] = val
                self.update(*args)
            return Signal(self.dom[key], index_signal, index_update)
        if isinstance(key, slice):
            indices = list(itertools.islice(range(len(self.dom)), key.start,
                                            key.stop, key.step))
            def slice_signal(*vals):
                args = [None for _ in range(len(self.dom))]
                for i, v in zip(indices, vals):
                    args[i] = v
                return self(*args)[key]
            def slice_update(*vals):
                args = [None for _ in range(len(self.dom))]
                for i, v in zip(indices, vals):
                    args[i] = v
                self.update(*args)
            return Signal(self.dom[key], slice_signal, slice_update)
        raise TypeError(messages.type_err((int, slice), key))

    def split(self):
        return tuple(self[i] for i in range(len(self.dom)))
