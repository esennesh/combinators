#!/usr/bin/env python3

from functools import reduce

from discopy import cartesian, cat, messages, monoidal, rigid
from discopy.monoidal import PRO, Sum, Ty

LENS_OB_DESCRIPTION = r'$\binom{%s}{%s}$'

class LensOb(cat.Ob):
    def __init__(self, upper=rigid.PRO(1), lower=rigid.PRO(1)):
        if not isinstance(upper, Ty):
            if isinstance(upper, cat.Ob):
                upper = upper.name
            upper = Ty(upper)
        self._upper = upper

        if not isinstance(lower, Ty):
            if isinstance(lower, cat.Ob):
                lower = lower.name
            lower = Ty(lower)
        self._lower = lower

        super().__init__(LENS_OB_DESCRIPTION % (repr(self._upper),
                                                repr(self._lower)))

    @property
    def upper(self):
        return self._upper

    @property
    def lower(self):
        return self._lower

    def __eq__(self, other):
        if not isinstance(other, LensOb):
            return False
        return self.upper == other.upper and self.lower == other.lower

    def __hash__(self):
        return hash(repr(self))

cat.Ob.__and__ = lambda self, other: LensTy(LensOb(self, other))

class LensTy(Ty):
    def __init__(self, *objects):
        assert all(isinstance(ob, LensOb) for ob in objects)
        super().__init__(*objects)

    @staticmethod
    def named(*names):
        return LensTy(*[LensOb(name, name + "-prime") for name in names])

    @staticmethod
    def upgrade(old):
        return LensTy(*old.objects)

    @property
    def upper(self):
        return reduce(lambda x, y: x @ y, [ob.upper for ob in self.objects],
                      Ty())

    @property
    def lower(self):
        return reduce(lambda x, y: x @ y, [ob.lower for ob in self.objects],
                      Ty())

class LensPRO(LensTy):
    def __init__(self, n=0):
        if isinstance(n, LensPRO):
            n = len(n)
        if isinstance(n, cat.Ob):
            n = 1
        lens_ob = LensOb(cat.Ob(1), cat.Ob(1))
        super().__init__(*(n * [lens_ob]))

    def __repr__(self):
        return "LensPRO({})".format(len(self))

class LensDiagram(monoidal.Diagram):
    """
    Implements diagrams of lenses composed of Python functions
    """
    def __init__(self, dom, cod, boxes, offsets, layers=None):
        assert isinstance(dom, LensTy)
        assert isinstance(cod, LensTy)
        super().__init__(dom, cod, boxes, offsets, layers=layers)

    def __call__(self, *vals, **kwargs):
        """
        Call method implemented using the lens Functor.
        """
        if kwargs:
            vals = vals + (kwargs,)
        return SAMPLE_FUNCTOR(self)(*vals)

    @staticmethod
    def upgrade(old):
        return LensDiagram(old.dom, old.cod, old.boxes, old.offsets, old.layers)

    @staticmethod
    def id(dom):
        return Id(dom)

class Id(LensDiagram):
    """
    Implements identity diagrams on dom inputs.
    """
    def __init__(self, dom):
        """
        >>> assert Diagram.id(42) == Id(42) == Diagram(42, 42, [], [])
        """
        assert isinstance(dom, LensTy)
        super().__init__(dom, dom, [], [], layers=None)

    def __repr__(self):
        """
        >>> Id(42)
        Id(42)
        """
        return "Id({})".format(self.dom)

    def __str__(self):
        """
        >>> print(Id(42))
        Id(42)
        """
        return repr(self)

class LensFunctor(monoidal.Functor):
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=LensTy, ar_factory=LensFunction)

class LensBox(monoidal.Box, LensDiagram):
    def __init__(self, name, dom, cod, sample, update, data={}):
        assert isinstance(dom, LensTy)
        assert isinstance(cod, LensTy)
        self._sample = sample
        self._update = update
        rigid.Box.__init__(self, name, dom, cod, data=data)
        LensDiagram.__init__(self, dom, cod, [self], [0])

    @property
    def sample(self):
        return self._sample

    @property
    def update(self):
        return self._update

class Unit(LensBox):
    def __init__(self, val, cod):
        def unit(val=val):
            return val
        super().__init__('Unit(%d)' % len(cod), LensPRO(0), cod, unit,
                         lambda fb: ())

class Copy(LensDiagram):
    """
    Implements the copy function from dom to 2*dom.
    >>> assert Copy(3)(0, 1, 2) == (0, 1, 2, 0, 1, 2)
    """
    def __init__(self, dom):
        if not isinstance(dom, LensTy):
            dom = LensPRO(dom)
        result = Id(0)
        for ob in dom:
            result = result @ _copy(ob)
        for i in range(1, len(dom)):
            swaps = Id(0).tensor(*[_swap(dom[k], dom[k+i]) for k in
                                   range(len(dom) - i)])
            # swaps = Id(0).tensor(*((len(dom) - i) * [SWAP]))
            result = result >> Id(dom[:i]) @ swaps @ Id(dom[-i:])
        super().__init__(dom, dom @ dom, result.boxes, result.offsets,
                         layers=result.layers)

class Swap(LensDiagram):
    def __init__(self, left, right):
        if not isinstance(left, LensTy):
            left = LensPRO(left)
        if not isinstance(right, LensTy):
            right = LensPRO(right)
        dom, cod = left @ right, right @ left
        boxes = [SWAP for i in range(len(left)) for j in range(len(right))]
        offsets = [left + i - 1 - j for j in range(len(left))
                   for i in range(len(right))]
        super().__init__(dom, cod, boxes, offsets)

def _copy(ob):
    assert isinstance(ob, LensOb)
    return LensBox('copy', LensTy(ob), LensTy(ob, ob),
                   lambda *vals: vals + vals, lambda x, y, feedback: feedback)

def _swap(obx, oby):
    assert isinstance(obx, LensOb) and isinstance(oby, LensOb)
    return LensBox('swap', LensTy(obx, oby), LensTy(oby, obx),
                   lambda x, y: (y, x), lambda x, y, fby, fbx: (fbx, fby))

COPY = LensBox('copy', LensPRO(1), LensPRO(2), lambda *vals: vals + vals,
               lambda x, y, feedback: feedback)
SWAP = LensBox('swap', LensPRO(2), LensPRO(2), lambda x, y: (y, x),
               lambda x, y, fby, fbx: (fbx, fby))

class LensFunction(monoidal.Box):
    def __init__(self, name, dom, cod, sample, update, **params):
        assert isinstance(dom, LensTy)
        assert isinstance(cod, LensTy)

        self._name = name
        self._sample = sample
        self._update = update

        super().__init__(str(sample), dom, cod, **params)

    def __call__(self, *vals):
        return self._sample(*vals)

    @property
    def sample(self):
        return self._sample

    @property
    def update(self):
        return self._update

    def then(self, *others):
        """
        Implements the sequential composition of lenses.
        """
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.then(self, *others)
        other = others[0]
        if not isinstance(other, LensFunction):
            raise TypeError(messages.type_err(LensFunction, other))
        if len(self.cod) != len(other.dom):
            raise cat.AxiomError(messages.does_not_compose(self, other))

        sample = self.sample >> other.sample
        update0 = cartesian.Copy(len(self.dom.upper)) @\
                  cartesian.Id(len(other.cod.lower))
        update1 = cartesian.Id(len(self.dom.upper)) @ self.sample @\
                  cartesian.Id(len(other.cod.lower))
        update2 = cartesian.Id(len(self.dom.upper)) @ other.update
        update = update0 >> update1 >> update2 >> self.update
        return LensFunction('%s >> %s' % (self.name, other.name), self.dom,
                            other.cod, sample, update)

    def tensor(self, *others):
        """
        Implements the tensor product of lenses.
        """
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.tensor(self, *others)
        other = others[0]
        if not isinstance(other, LensFunction):
            raise TypeError(messages.type_err(LensFunction, other))

        dom = self.dom @ other.dom
        cod = self.cod @ other.cod

        sample = self.sample @ other.sample
        update = self.update @ other.update
        return LensFunction('%s @ %s' % (self.name, other.name), dom, cod,
                            sample, update)

    @staticmethod
    def id(dom):
        assert isinstance(dom, LensTy)
        sample = cartesian.Id(len(dom.upper))
        update = cartesian.Discard(len(dom)) @ cartesian.Id(len(dom.lower))
        return LensFunction('Id(%d)' % len(dom.upper), dom, dom, sample, update)

    @staticmethod
    def create(box):
        sample = cartesian.Box(box.name + '_sample', len(box.dom.upper),
                               len(box.cod.upper), box.sample)
        update = cartesian.Box(box.name + '_update',
                               len(box.dom.upper @ box.cod.lower),
                               len(box.dom.lower), box.update)
        return LensFunction(box.name, box.dom, box.cod, sample, update)

SAMPLE_FUNCTOR = LensFunctor(lambda lob: lob, LensFunction.create)
