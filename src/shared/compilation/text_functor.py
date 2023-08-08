from enum import Enum
from discopy import rigid


class TextFunctor(Enum):
    """Enum to distinguish various ways of expanding higher order boxes"""
    Id = "id"
    RemoveThe = "remove-the"
    RemoveNames = "remove-names"

    def __repr__(self):
        return self._name_


def ar_remove_the(a):
    if isinstance(a, rigid.Box) and a.name in ["the", "The"]:
        return rigid.Id(a.dom)
    return a


def ar_remove_names(a):
    if isinstance(a, rigid.Box) and a.name in ["Alice", "Bob", "Charlie", "John", "Mary", "Sam"]:  # todo, hardcoded for now
        return rigid.Box("INIT_0", a.dom, a.cod)
    return a


_convert = {
    TextFunctor.RemoveThe: rigid.Functor(ob=lambda x: x, ar=ar_remove_the),
    TextFunctor.RemoveNames: rigid.Functor(ob=lambda x: x, ar=ar_remove_names),
    TextFunctor.Id: rigid.Functor(ob=lambda x: x, ar=lambda b: b),  # pass through
}


def get_text_functor_from_config(config):
    return _convert[config["text_functor"]]
