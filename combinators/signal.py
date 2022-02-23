#!/usr/bin/env python3

from functools import wraps
import itertools

from discopy import cartesian, messages, monoidal

from . import lens

class Signal:
    """
    Wraps Python functions that can be sliced into separate wires, each of which
    may have a cached default value, with domain and codomain information.
    """
    def __init__(self, dom, function, update):
        self._dom = dom
        assert callable(function)
        self._function = function
        assert callable(update)
        self._update = update

    def __repr__(self):
        return "Signal(dom={}, function={}, update={})".format(
            self.dom, repr(self._function), repr(self._update))

    def __str__(self):
        return repr(self)

    @property
    def dom(self):
        return self._dom

    @property
    def update(self):
        return self._update

    def __call__(self):
        return cartesian.tuplify(self._function())

    def __getitem__(self, key):
        if isinstance(key, int):
            @wraps(self._function)
            def index_signal():
                return self._function()[key]
            @wraps(self._update)
            def index_update(val):
                args = [None for _ in range(self.dom)]
                args[key] = val
                self.update(*args)
            return Signal(1, index_signal, index_update)
        if isinstance(key, slice):
            indices = list(itertools.islice(range(self.dom), key.start,
                                            key.stop, key.step))
            def slice_signal():
                return self._function()[key]
            def slice_update(*vals):
                args = [None for _ in range(self.dom)]
                for i, v in zip(indices, vals):
                    args[i] = v
                self.update(*args)
            return Signal(key.stop-key.start, slice_signal, slice_update)
        raise TypeError(messages.type_err((int, slice), key))

    def split(self):
        return tuple(self[i] for i in range(self.dom))

class Cap(lens.Cap):
    def cap_put(self, *_):
        def cap_signal():
            return self._vals
        def cap_update(*_):
            pass
        return Signal(len(self.dom), cap_signal, cap_update).split()
