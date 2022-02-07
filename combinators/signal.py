#!/usr/bin/env python3

from abc import abstractmethod
from functools import reduce, wraps
import itertools

from discopy import cartesian, cat, messages, monoidal, rigid

class Signal(rigid.Box):
    """
    Wraps Python functions that can be sliced into separate wires, each of which
    may have a cached default value, with domain and codomain information.
    """
    def __init__(self, dom, cod, function):
        self._function = function
        super().__init__(repr(function), monoidal.PRO(dom), monoidal.PRO(cod))

    def __repr__(self):
        return "Signal(dom={}, cod={}, function={})".format(
            self.dom, self.cod, repr(self._function))

    def __str__(self):
        return repr(self)

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    @staticmethod
    def id(dom=0):
        return Signal(dom, dom, cartesian.untuplify)

    def then(self, *others):
        assert len(others) == 1
        other = others[0]
        if not isinstance(other, Signal):
            raise TypeError(messages.type_err(Signal, other))
        if len(self.cod) != len(other.dom):
            raise cat.AxiomError(messages.does_not_compose(self, other))

        function = lambda vals: other(*cartesian.tuplify(self(*vals)))
        return Signal(self.dom, other.cod, function)

    def tensor(self, *others):
        assert len(others) == 1
        other = others[0]
        if not isinstance(other, Signal):
            raise TypeError(messages.type_err(Signal, other))
        dom, cod = self.dom @ other.dom, self.cod @ other.cod

        def product(*vals):
            vals0 = cartesian.tuplify(self(*vals[:len(self.dom)]))
            vals1 = cartesian.tuplify(other(*vals[len(self.dom):]))
            return cartesian.untuplify(*(vals0 + vals1))
        return Signal(dom, cod, product)

    def __getitem__(self, key):
        if isinstance(key, int):
            def index_signal(val):
                args = [None for _ in range(len(self.dom))]
                args[key] = val
                return self(*args)[key]
            return Signal(self.dom[key], self.cod[key], index_signal)
        if isinstance(key, slice):
            indices = list(itertools.islice(range(len(self.dom)), key.start,
                                            key.stop, key.step))
            def slice_signal(*vals):
                args = [None for _ in range(len(self.dom))]
                for i, v in zip(indices, vals):
                    args[i] = v
                return self(*args)[key]
            return Signal(self.dom[key], self.cod[key], slice_signal)
        raise TypeError(messages.type_err((int, slice), key))

    def split(self):
        return tuple(self[i] for i in range(len(self.dom)))
