#!/usr/bin/env python3

import collections

from .. import lens, sampler

class SequentialConditioner(lens.Functor):
    def __init__(self, **kwargs):
        super().__init__(lambda lob: lob, self._condition)
        self._boxes = {k: collections.deque(v) for k, v in kwargs.items()}

    def _condition(self, box):
        if box.name in self._boxes:
            assert isinstance(box, sampler.ImportanceBox)

            data = self._boxes[box.name].popleft()
            if not isinstance(data, dict):
                data = {'data': data}

            return box.__class__(box.name, box.dom, box.cod, box.target,
                                 box.proposal, data={**data, **box.data})
        return box
