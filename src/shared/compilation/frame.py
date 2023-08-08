from enum import Enum
from discopy import rigid
from discocirc.diag.frame import Frame as DefaultFrame


def _decompose_pass_nouns(self):
    """Decompose higher order boxes passing all nouns through, plus an extra connecting wire"""
    s = rigid.Ty('n')
    w = rigid.Id(s)
    if len(self.dom) == 1:
        inside_dom = rigid.Ty().tensor(
            *[b.dom @ s for b in self._insides])
        inside_cod = rigid.Ty().tensor(
            *[b.cod @ s for b in self._insides])
        inside = [(DefaultFrame.get_decompose_functor())(b) @ w
                for b in self._insides]
        top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
        bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
        mid = rigid.Id().tensor(*inside)

    elif len(self.dom) == 2:
        inside_dom = rigid.Ty().tensor(
            *[b.dom @ s for b in self._insides]) @ s
        inside_cod = rigid.Ty().tensor(
            *[b.cod @ s for b in self._insides]) @ s
        inside = [(DefaultFrame.get_decompose_functor())(b) @ w
                for b in self._insides]
        top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
        bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
        mid = rigid.Id().tensor(*inside) @ w
        # equation(top, mid, bot)

    return top >> mid >> bot


def _decompose_single_r(self):
    """Decompose higher order boxes with just one connecting * wire on the right."""
    s = rigid.Ty('*')
    w = rigid.Id(s)

    inside_dom = rigid.Ty().tensor(
        *[b.dom for b in self._insides]) @ s
    inside_cod = rigid.Ty().tensor(
        *[b.cod for b in self._insides]) @ s
    inside = [(DefaultFrame.get_decompose_functor())(b)
              for b in self._insides]
    top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
    bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
    mid = rigid.Id().tensor(*inside) @ w
    # equation(top, mid, bot)
    return top >> mid >> bot


def _decompose_disconnected(self):
    """Decompose higher order boxes no connections between top and bottom boxes."""
    inside_dom = rigid.Ty().tensor(
        *[b.dom for b in self._insides])
    inside_cod = rigid.Ty().tensor(
        *[b.cod for b in self._insides])
    inside = [(DefaultFrame.get_decompose_functor())(b)
              for b in self._insides]
    top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
    bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
    mid = rigid.Id().tensor(*inside)
    # equation(top, mid, bot)
    return top >> mid >> bot


def _decompose_quantum(self):
    """Decompose higher order boxes deleting and re-initialising all noun wires not involved internally.
    Also adds a connecting wire."""
    n = rigid.Ty('n')
    s = rigid.Ty('*')
    w = rigid.Id(s)
    n_extras_in = len(self.dom) - sum([len(inside.dom) for inside in self._insides])
    n_extras_out = len(self.cod) - sum([len(inside.cod) for inside in self._insides])

    inside_dom = rigid.Ty().tensor(
        *[b.dom for b in self._insides], *[n for _ in range(n_extras_in)]) @ s
    inside_cod = rigid.Ty().tensor(
        *[b.cod for b in self._insides], *[n for _ in range(n_extras_out)]) @ s
    inside = [
        (DefaultFrame.get_decompose_functor())(b)
        for b in self._insides
    ]
    reint = rigid.Id().tensor(*[
        rigid.Box("DISCARD", n, rigid.Ty()) for _ in range(n_extras_in)
    ]) >> rigid.Id().tensor(*[
        rigid.Box("INIT_0", rigid.Ty(), n) for _ in range(n_extras_out)
    ])
    top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
    bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
    mid = rigid.Id().tensor(*inside) @ reint @ w

    return top >> mid >> bot


def _decompose_no_discards(self):
    """Decompose higher order boxes for pure quantum circuits."""
    n = rigid.Ty('n')
    s = rigid.Ty('*')
    w = rigid.Id(s)
    n_extras_in = len(self.dom) - sum([len(inside.dom) for inside in self._insides])
    n_extras_out = len(self.cod) - sum([len(inside.cod) for inside in self._insides])
    assert n_extras_out == n_extras_in  # Can only avoid discards if this is the case

    inside_dom = rigid.Ty().tensor(
        *[b.dom for b in self._insides], *[n for _ in range(n_extras_in)])
    inside_cod = rigid.Ty().tensor(
        *[b.cod for b in self._insides], *[n for _ in range(n_extras_out)])
    inside = [
        (DefaultFrame.get_decompose_functor())(b)
        for b in self._insides
    ]
    extras = rigid.Id().tensor(*[
        rigid.Id(n) for _ in range(n_extras_in)
    ])
    top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
    bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
    mid = rigid.Id().tensor(*inside) @ extras

    return top >> mid >> bot


def _decompose_name_only(self):
    """Only decompose the box name."""
    inside_dom = rigid.Ty().tensor(
        *[b.dom for b in self._insides])
    inside_cod = rigid.Ty().tensor(
        *[b.cod for b in self._insides])
    inside = [(DefaultFrame.get_decompose_functor())(b)
              for b in self._insides]
    top = rigid.Box(f'[{self.name}]', self.dom, inside_dom)
    bot = rigid.Box(f'[\\{self.name}]', inside_cod, self.cod)
    mid = rigid.Id().tensor(*inside)
    # equation(top, mid, bot)
    return rigid.Box(self.drawing_name, self.dom, self.cod)


class DecomposeFrame(Enum):
    """Enum to distinguish various ways of expanding higher order boxes"""
    NoDecomp = ""
    Default = "default"
    SingleConnectionR = "single-connection-r"
    Disconnected = "disconnected"
    PassNouns = "pass-nouns"
    Quantum = "quantum"
    NoDiscards = "no-discards"
    NameOnly = "name-only"

    def __repr__(self):
        return self._name_


_convert = {
    DecomposeFrame.SingleConnectionR: _decompose_single_r,
    DecomposeFrame.Disconnected: _decompose_disconnected,
    DecomposeFrame.PassNouns: _decompose_pass_nouns,
    DecomposeFrame.NameOnly: _decompose_name_only,
    DecomposeFrame.Quantum: _decompose_quantum,
    DecomposeFrame.NoDiscards: _decompose_no_discards,
    DecomposeFrame.NoDecomp: lambda self: self,  # pass through
}


class FrameFunctor(rigid.Functor):
    """
    Convert diagram frames into a different type (with different decomposition)
    """

    def __init__(self, ob, ar, ob_factory=rigid.Ty, ar_factory=rigid.Diagram, frame=DefaultFrame):
        super().__init__(ob, ar, ob_factory=ob_factory, ar_factory=ar_factory)
        self.frame_factory = frame

    def __call__(self, diagram):
        # Only need to do something special if we come across a frame
        if isinstance(diagram, DefaultFrame):
            # Recursively convert nested frames
            return self.frame_factory(
                diagram.name,
                diagram.dom,
                diagram.cod,
                insides=[self.__call__(diag) for diag in diagram.insides]
            )
        return super().__call__(diagram)


def create_frame(decomp: DecomposeFrame):
    # Return a frame that decomposes according to the given function.
    if decomp == DecomposeFrame.Default:
        return DefaultFrame

    decompose = _convert[decomp]

    class CustomFrame(DefaultFrame):
        def __init__(self, name, dom, cod, insides):
            super().__init__(name, dom, cod, insides)

        def _decompose(self):
            return decompose(self)

    return CustomFrame


def get_conversion_functor(new_frame: DecomposeFrame):
    NewFrame = create_frame(new_frame)
    return FrameFunctor(
        ob=lambda x: x,
        ar=lambda b: b,
        frame=NewFrame,
    )


frame_expansion_functor = DefaultFrame.get_decompose_functor()
