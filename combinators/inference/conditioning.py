#!/usr/bin/env python3

import collections

from .. import lens, tracing

class SequentialConditioner(lens.LensFunctor):
    def __init__(self, **vals):
        super().__init__(lambda lob: lob, self._condition)
        self._boxes = {k: collections.deque(v) for k, v in vals.items()}

    def _condition(self, box):
        if isinstance(box, tracing.TracedLensBox) and box.name in self._boxes:
            return box.conditioned(self._boxes[box.name].popleft())
        return box
