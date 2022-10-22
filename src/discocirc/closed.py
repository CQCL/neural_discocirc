# -*- coding: utf-8 -*-

"""
Implements the free closed monoidal category.
"""

from discopy import monoidal, biclosed


class Ty(monoidal.Ty):
    """
    Objects in a free closed monoidal category.
    Generated by the following grammar:

        ty ::= Ty(name) | ty @ ty | ty >> ty

    Examples
    --------
    >>> x, y = Ty('x'), Ty('y')
    >>> print(x >> y >> x)
    ((x → y) → x)
    >>> print((y >> x >> y) @ x)
    ((y → x) → y) @ x
    """

    def __init__(self, *objects, left=None, right=None):
        self.left, self.right = left, right
        super().__init__(*objects)

    def __rshift__(self, other):
        return Func(self, other)


class Func(Ty):
    """ Function types. """
    def __init__(self, input=None, output=None):
        Ty.__init__(self, self)
        self.input, self.output = input, output

    def __repr__(self):
        return "Func({}, {})".format(repr(self.input), repr(self.output))

    def __str__(self):
        return "({} → {})".format(self.input, self.output)

    def __eq__(self, other):
        if not isinstance(other, Func):
            return False
        return self.input == other.input and self.output == other.output

    def __hash__(self):
        return hash(repr(self))


def biclosed_to_closed(x):
    """Converts the biclosed types to closed types."""
    if isinstance(x, biclosed.Under):
        fx = Func(biclosed_to_closed(x.left), biclosed_to_closed(x.right))
    elif isinstance(x, biclosed.Over):
        fx = Func(biclosed_to_closed(x.right), biclosed_to_closed(x.left))
    elif isinstance(x, biclosed.Ty):
        fx = Ty(*[biclosed_to_closed(y) for y in x.objects])
    else:
        fx = x
    return fx