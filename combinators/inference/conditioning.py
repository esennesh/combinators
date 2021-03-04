#!/usr/bin/env python3

import collections

from .. import lens, sampler

class SequentialConditioner(lens.LensFunctor):
    def __init__(self, *vals):
        super().__init__(lambda lob: lob, self._condition)
        self._vals = collections.deque(vals)

    def _condition(self, box):
        if isinstance(box.sample, sampler.ImportanceSampler):
            box.sample.condition(self._vals.popleft())
        return box
