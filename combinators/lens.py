#!/usr/bin/env python3

from abc import abstractmethod
from functools import reduce, wraps

from discopy import cartesian, cat, messages, monoidal, wiring

LENS_OB_DESCRIPTION = r'$\binom{%s}{%s}$'

class Ob(cat.Ob):
    def __init__(self, upper=monoidal.PRO(1), lower=monoidal.PRO(1)):
        if not isinstance(upper, cat.Ob):
            upper = cat.Ob(str(upper))
        self._upper = upper

        if not isinstance(lower, cat.Ob):
            lower = cat.Ob(str(lower))
        self._lower = lower

        super().__init__(LENS_OB_DESCRIPTION % (self.upper.name,
                                                self.lower.name))

    @property
    def upper(self):
        return self._upper

    @property
    def lower(self):
        return self._lower

    def __eq__(self, other):
        if not isinstance(other, Ob):
            return False
        return self.upper == other.upper and self.lower == other.lower

    def __hash__(self):
        return hash(repr(self))

cat.Ob.__and__ = Ob

class Ty(monoidal.Ty):
    def __init__(self, *objects):
        assert all(isinstance(ob, Ob) for ob in objects)
        super().__init__(*objects)

    @staticmethod
    def named(*names):
        return Ty(*[Ob(name, name + "-prime") for name in names])

    @staticmethod
    def upgrade(old):
        return Ty(*old.objects)

    @property
    def upper(self):
        return reduce(lambda x, y: x @ y, [ty(ob.upper) for ob in self.objects],
                      monoidal.Ty())

    @property
    def lower(self):
        return reduce(lambda x, y: x @ y, [ty(ob.lower) for ob in self.objects],
                      monoidal.Ty())

def ty(t):
    assert isinstance(t, cat.Ob)
    if isinstance(t, monoidal.Ty):
        return t
    return monoidal.Ty(t)

def lens_type(uppers, lowers):
    assert isinstance(uppers, monoidal.Ty) and isinstance(lowers, monoidal.Ty)
    uppers, lowers = uppers.objects, lowers.objects
    if len(uppers) > len(lowers):
        lowers = lowers + [monoidal.PRO(0) for _ in
                           range(len(uppers) - len(lowers))]
    elif len(lowers) > len(uppers):
        uppers = uppers + [monoidal.PRO(0) for _ in
                           range(len(lowers) - len(uppers))]

    return Ty(*[Ob(u, l) for u, l in zip(uppers, lowers)])

monoidal.Ty.__and__ = lens_type

class PRO(Ty):
    def __init__(self, n=0):
        if isinstance(n, PRO):
            n = len(n)
        if isinstance(n, cat.Ob):
            n = 1
        lens_ob = Ob(cat.Ob(1), cat.Ob(1))
        super().__init__(*(n * [lens_ob]))

    def __repr__(self):
        return "lens.PRO({})".format(len(self))

class CartesianSemanticsFunctor(wiring.Functor):
    def __init__(self):
        super().__init__(lambda t: t, self.semantics)

    @classmethod
    def semantics(cls, f):
        return CartesianWiringBox(f.name, f.dom, f.cod, f.getf, f.putf,
                                  data=f.data)

@monoidal.Diagram.subclass
class Diagram(monoidal.Diagram):
    """
    Implements diagrams of lenses composed of Python functions
    """
    CARTESIAN_SEMANTICS = CartesianSemanticsFunctor()
    CARTESIAN_GET = wiring.Functor(lambda t: monoidal.PRO(len(t)),
                                   lambda f: f.get(), monoidal.PRO,
                                   cartesian.Diagram)

    def __init__(self, dom, cod, boxes, offsets, layers=None):
        assert isinstance(dom, Ty)
        assert isinstance(cod, Ty)
        super().__init__(dom, cod, boxes, offsets, layers=layers)

    def __call__(self, *vals, **kwargs):
        """
        Get method for Cartesian lenses.
        """
        if kwargs:
            vals = vals + (kwargs,)
        get = Diagram.CARTESIAN_GET(Diagram.CARTESIAN_SEMANTICS(self))
        return get(*vals)

    def put(self, *vals, **kwargs):
        """
        Put method for Cartesian lenses.
        """
        if kwargs:
            vals = vals + (kwargs,)
        put = Diagram.CARTESIAN_SEMANTICS(self).collapse(__put_falg__)
        return put(*vals)

    @staticmethod
    def id(dom=Ty()):
        return Id(dom)

class Id(Diagram):
    """
    Implements identity diagrams on dom inputs.
    """
    def __init__(self, dom):
        """
        >>> assert Diagram.id(42) == Id(42) == Diagram(42, 42, [], [])
        """
        assert isinstance(dom, Ty)
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

class Functor(monoidal.Functor):
    def __init__(self, ob, ar):
        super().__init__(ob, ar, ob_factory=Ty, ar_factory=Diagram)

class Box(monoidal.Box, Diagram):
    def __init__(self, name, dom, cod, data={}):
        if not isinstance(dom, Ty):
            raise TypeError(messages.type_err(Ty, dom))
        if not isinstance(cod, Ty):
            raise TypeError(messages.type_err(Ty, cod))
        monoidal.Box.__init__(self, name, dom, cod, data=data)
        Diagram.__init__(self, dom, cod, [self], [0])

class CartesianBox(Box):
    def __init__(self, name, dom, cod, getf, putf, data={}):
        assert callable(getf) and callable(putf)
        self._getf = getf
        self._putf = putf
        super().__init__(name, dom, cod, data=data)

    @property
    def getf(self):
        return self._getf

    @property
    def putf(self):
        return self._putf

class Unit(CartesianBox):
    def __init__(self, val, cod):
        def unit(val=val):
            return val
        super().__init__('Unit(%d)' % len(cod), PRO(0), cod, unit,
                         lambda *args: ())

class Copy(Diagram):
    """
    Implements the copy function from dom to 2*dom.
    >>> assert Copy(3)(0, 1, 2) == (0, 1, 2, 0, 1, 2)
    """
    def __init__(self, dom):
        if not isinstance(dom, Ty):
            dom = PRO(dom)
        tensor_id = Id(PRO(0))
        result = tensor_id
        for ob in dom:
            result = result @ _copy(ob)
        for i in range(1, len(dom)):
            swaps = tensor_id.tensor(*[_swap(dom[k], dom[k+i]) for k in
                                       range(len(dom) - i)])
            result = result >> Id(dom[:i]) @ swaps @ Id(dom[-i:])
        super().__init__(dom, dom @ dom, result.boxes, result.offsets,
                         layers=result.layers)

class Swap(Diagram):
    def __init__(self, left, right):
        if not isinstance(left, Ty):
            left = PRO(left)
        if not isinstance(right, Ty):
            right = PRO(right)
        dom, cod = left @ right, right @ left
        boxes = [SWAP for i in range(len(left)) for j in range(len(right))]
        offsets = [left + i - 1 - j for j in range(len(left))
                   for i in range(len(right))]
        super().__init__(dom, cod, boxes, offsets)

class Discard(Diagram):
    def __init__(self, dom):
        if not isinstance(dom, Ty):
            dom = PRO(dom)
        result = Id(PRO(0)).tensor(*(len(dom) * [DISCARD]))
        super().__init__(result.dom, result.cod, result.boxes, result.offsets,
                         layers=result.layers)

def _copy(ob):
    assert isinstance(ob, Ob)
    return CartesianBox('copy', Ty(ob), Ty(ob, ob), lambda *vals: vals + vals,
                        lambda x, y, feedback: feedback)

def _swap(obx, oby):
    assert isinstance(obx, Ob) and isinstance(oby, Ob)
    return CartesianBox('swap', Ty(obx, oby), Ty(oby, obx), lambda x, y: (y, x),
                        lambda x, y, fby, fbx: (fbx, fby))

COPY = CartesianBox('copy', PRO(1), PRO(2), lambda *vals: vals + vals,
                    lambda x, y, feedback: feedback)
SWAP = CartesianBox('swap', PRO(2), PRO(2), lambda x, y: (y, x),
                    lambda x, y, fby, fbx: (fbx, fby))

DISCARD = CartesianBox('discard', monoidal.PRO(1) & monoidal.PRO(0), PRO(0),
                       lambda *x: (), lambda p, *x: ((), p))

# TODO: rewrite this for the new get-put framework, once I've invented one
def hook(lens, pre_sample=None, pre_update=None, post_sample=None,
         post_update=None):
    sample = lens.sample
    if pre_sample:
        sample = _prehook_method(sample, pre_sample)
    if post_sample:
        sample = _posthook_method(sample, post_sample)

    update = lens.update
    if pre_update:
        update = _prehook_method(update, pre_update)
    if post_update:
        update = _posthook_method(update, post_update)

    lens.sample = sample
    lens.update = update

    return lens

def _prehook_method(method, h):
    @wraps(method)
    def m(self, *args, **kwargs):
        args, kwargs = h(method.__self__, *args, **kwargs)
        return method(*args, **kwargs)
    m.__self__ = method.__self__
    return m

def _posthook_method(method, h):
    @wraps(method)
    def m(*args, **kwargs):
        vals = method(*args, **kwargs)
        return h(method.__self__, vals)
    m.__self__ = method.__self__
    return m

class WiringBox(wiring.Box):
    def __init__(self, name, dom, cod, **params):
        if not isinstance(dom, Ty):
            raise TypeError(messages.type_err(Ty, dom))
        if not isinstance(cod, Ty):
            raise TypeError(messages.type_err(Ty, cod))
        super().__init__(name, dom, cod, **params)

    @abstractmethod
    def get(self) -> monoidal.Box:
        pass

    @abstractmethod
    def put(self) -> monoidal.Box:
        pass

class CartesianWiringBox(WiringBox):
    def __init__(self, name, dom, cod, getf, putf, **params):
        self._getf = getf
        self._putf = putf
        super().__init__(name, dom, cod, **params)

    def get(self):
        dom = monoidal.PRO(len(self.dom.upper))
        cod = monoidal.PRO(len(self.cod.upper))
        return cartesian.Box(self.name + '_get', dom, cod, function=self._getf,
                             data=self.data)

    def put(self):
        dom = monoidal.PRO(len(self.dom.upper @ self.cod.lower))
        cod = monoidal.PRO(len(self.dom.lower))
        return cartesian.Box(self.name + '_put', dom, cod, function=self._putf,
                             data=self.data)

def __put_falg__(f):
    if isinstance(f, wiring.Id):
        discard = cartesian.Discard(monoidal.PRO(len(f.dom.upper)))
        ident = cartesian.Id(monoidal.PRO(len(f.cod.lower)))
        return cartesian.Id(monoidal.PRO(len(f.dom.upper))), discard @ ident
    if isinstance(f, wiring.Box):
        assert isinstance(f, CartesianWiringBox)
        return f.get(), f.put()
    if isinstance(f, wiring.Sequential):
        def put_compose(f, g):
            f_upper = len(f[0].dom)
            g_lower = len(g[1].dom) - len(g[0].dom)
            put = cartesian.Copy(f_upper) @ cartesian.Id(g_lower)
            put = put >> (cartesian.Id(f_upper) @ f[0] @ cartesian.Id(g_lower))
            put = put >> (cartesian.Id(f_upper) @ g[1])
            put = put >> f[1]
            return (f[0] >> g[0]), put
        return reduce(put_compose, f.arrows)
    if isinstance(f, wiring.Parallel):
        def put_tensor(f, g):
            f_upper = len(f[0].dom)
            f_lower = len(f[1].dom) - f_upper

            g_upper = len(g[0].dom)
            g_lower = len(g[1].dom) - g_upper

            put = cartesian.Id(f_upper) @ cartesian.Swap(g_upper, f_lower) @\
                  cartesian.Id(g_lower)
            return put >> (f[1] @ g[1])
        return reduce(put_tensor, f.factors)
    raise TypeError(messages.type_err(wiring.Wiring, f))

def getter(diagram):
    return Diagram.CARTESIAN_GET(diagram)

def putter(diagram):
    return diagram.collapse(__put_falg__)[1]
