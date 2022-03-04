#!/usr/bin/env python3

from abc import abstractmethod
from functools import lru_cache, reduce, wraps

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
    FUNCTION_SEMANTICS = cartesian.PythonFunctor(lambda t: monoidal.PRO(len(t)),
                                                 lambda f: cartesian.Function(
                                                     len(f.dom), len(f.cod),
                                                     f.function
                                                 ))

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

class Cap(CartesianBox):
    def __init__(self, dom):
        assert isinstance(dom, Ty)
        self._vals = (None,) * len(dom)
        super().__init__('Cap(%d)' % len(dom), dom, PRO(0), self.cap_get,
                         self.cap_put)

    def cap_get(self, *args):
        self._vals = args
        return ()

    def cap_put(self, *_):
        return tuple(reversed(self._vals))

class Copy(CartesianBox):
    def __init__(self, dom, n=2, join=None):
        self._join = join if join else lambda x, y: (x, y)
        self._n = n
        cod = reduce(lambda x, y: x @ y, [dom] * self._n)
        super().__init__('Copy(%d, %d)' % (len(dom), self._n), dom, cod,
                         self.copy, self.combine)

    def copy(self, *args):
        return args * self._n

    def combine(self, *args):
        assert len(args) == len(self.dom) + len(self.cod)
        bkwds = args[len(self.dom):len(self.dom) + len(self.cod)]
        for k in range(1, self._n):
            bkwdy = args[(k + 1) * len(self.dom):(k + 2) * len(self.dom)]
            bkwds = tuple(self._join(x, y) for x, y in zip(bkwds, bkwdy))
        return bkwds

class Swap(CartesianBox):
    def __init__(self, left, right):
        if not isinstance(left, Ty):
            left = PRO(left)
        if not isinstance(right, Ty):
            right = PRO(right)

        self._left = left
        self._right = right
        super().__init__('Swap(%d, %d)' % (len(left), len(right)), left @ right,
                         right @ left, self.swapf, self.swapb)

    def swapf(self, *args):
        left = args[:len(self._left)]
        right = args[len(self._left):]
        return (*right, *left)

    def swapb(self, *args):
        right = args[:len(self._right)]
        left = args[len(self._right):]
        return (*left, *right)

class Discard(Diagram):
    def __init__(self, dom):
        if not isinstance(dom, Ty):
            dom = PRO(dom)
        result = Id(PRO(0)).tensor(*(len(dom) * [DISCARD]))
        super().__init__(result.dom, result.cod, result.boxes, result.offsets,
                         layers=result.layers)

DISCARD = CartesianBox('discard', PRO(1), PRO(0), lambda x: (),
                       lambda x: None)

def hook(lens, pre_get=None, pre_put=None, post_get=None, post_put=None):
    getf = lens._getf
    if pre_get:
        getf = _prehook_method(getf, pre_get)
    if post_get:
        getf = _posthook_method(getf, post_get)

    putf = lens._putf
    if pre_put:
        putf = _prehook_method(putf, pre_put)
    if post_put:
        putf = _posthook_method(putf, post_put)

    lens._getf = getf
    lens._putf = putf

    return lens

def _prehook_method(method, h):
    @wraps(method)
    def m(*args, **kwargs):
        args, kwargs = h(*args, **kwargs)
        return method(*args, **kwargs)
    m.__self__ = method.__self__
    return m

def _posthook_method(method, h):
    @wraps(method)
    def m(*args, **kwargs):
        vals = method(*args, **kwargs)
        vals, _ = h(*vals)
        return vals
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
        assert isinstance(dom, Ty)
        assert isinstance(cod, Ty)
        super().__init__(name, dom, cod, **params)

    def get(self):
        dom = len(self.dom.upper)
        cod = len(self.cod.upper)
        return cartesian.Box(self.name + '_get', dom, cod, function=self._getf,
                             data=self.data)

    def put(self):
        dom = len(self.dom.upper @ self.cod.lower)
        cod = len(self.dom.lower)
        return cartesian.Box(self.name + '_put', dom, cod, function=self._putf,
                             data=self.data)

def __put_falg__(f):
    if isinstance(f, wiring.Id):
        dom = Ty(*f.dom.objects)
        discard = cartesian.Discard(len(dom.upper))
        ident = cartesian.Id(len(dom.lower))
        return cartesian.Id(len(dom.upper)), discard @ ident
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
            return (f[0] @ g[0]), put >> (f[1] @ g[1])
        return reduce(put_tensor, f.factors)
    raise TypeError(messages.type_err(wiring.Diagram, f))

@lru_cache(maxsize=None)
def getter(diagram):
    get = Diagram.CARTESIAN_GET(diagram)
    return Diagram.FUNCTION_SEMANTICS(get)

@lru_cache(maxsize=None)
def putter(diagram):
    _, put = diagram.collapse(__put_falg__)
    return Diagram.FUNCTION_SEMANTICS(put)
