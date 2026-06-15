from .representation import Representation
from abc import ABC, abstractmethod


class AnalysisObject(ABC):
    def __init__(self, repr=None, **kwargs):
        if repr is not None:
            assert isinstance(repr, Representation)
            self.repr = repr
        super().__init__(**kwargs)

    @property
    @abstractmethod
    def semantics(self):
        raise NotImplementedError

    def _store_kwargs(self, kwargs: dict):
        if not hasattr(self, '_params'):
            self._params = {}
        self._params.update(kwargs)

    def plot(self, *, ax=None, **kwargs):
        raise NotImplementedError

    def to_nix(self):
        raise NotImplementedError


class _MatrixAnalysisObject(AnalysisObject):
    """Proxy base for AnalysisObjects whose repr is a 2D matrix-like Representation.

    Subclasses declare their own instance attributes in _own_attrs so the
    proxy knows which names to store locally vs. forward to repr.
    """

    _own_attrs: frozenset = frozenset()

    def __setattr__(self, name, value):
        own = {'repr', 'semantics', 'to_nix', 'plot'} | type(self)._own_attrs
        if name in own or 'repr' not in object.__getattribute__(self, '__dict__'):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, 'repr'), name, value)

    def __getattribute__(self, name):
        own = (
            {'__class__', '__dict__', '__module__', '__repr__',
             '__setattr__', '__getattribute__', '__str__', '__init__',
             'repr', 'semantics', 'to_nix', 'plot', '_own_attrs'}
            | type(self)._own_attrs
        )
        if name in own:
            return object.__getattribute__(self, name)
        if 'repr' in object.__getattribute__(self, '__dict__'):
            repr_obj = object.__getattribute__(self, 'repr')
            if hasattr(repr_obj, name):
                return getattr(repr_obj, name)
        return object.__getattribute__(self, name)

    def __str__(self):
        return self.repr.__str__()

    def __len__(self):
        return len(self.repr)

    def __getitem__(self, key):
        return self.repr[key]

    def __array__(self, dtype=None):
        import numpy as np
        return np.asarray(self.repr, dtype=dtype)

    def plot(self, *, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        im = ax.imshow(self.repr.magnitude, **kwargs)
        ax.set_title(self.semantics)
        ax.set_xlabel("Neuron index")
        ax.set_ylabel("Neuron index")
        plt.colorbar(im, ax=ax)
        return ax

    def to_nix(self, nix_file_or_path, **kwargs):
        return self.repr.to_nix(nix_file_or_path=nix_file_or_path, **kwargs)