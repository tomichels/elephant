import matplotlib.pyplot as plt
import quantities as pq
from elephant.objects.base import AnalysisObject
from elephant.objects.representation import Representation
from elephant.objects.timeseries import TimeSeriesObject


class RateObject(AnalysisObject):
    def __init__(
        self, repr: Representation, kernel: str | None = None, method: str | None = None
    ):
        if not isinstance(repr, Representation):
            raise TypeError(
                f"""
                Representation of {self.__class__.__name__} has
                to be a {TimeSeriesObject.__name__}-Object"""
            )
        super().__init__(repr)

        self.method = method
        # print(self.repr.__dict__)
        self.kernel = kernel

    @property
    def semantics(self):
        if hasattr(self, "method"):
            return f"Firing rate with method {self.method}"
        return "Firing rate"
    
    def __setattr__(self, name, value):
        if name in ['repr', 'method', 'kernel', 'semantics', 'to_nix']:
            object.__setattr__(self, name, value)            
        else:
            repr_obj = object.__getattribute__(self, "repr")
            setattr(repr_obj, name, value)

    def __getattribute__(self, name):
        local_attributes = {
        '__class__', '__dict__', '__module__', '__repr__',
        '__setattr__', '__getattribute__', '__str__', '__init__',
        'repr', 'method', 'kernel', 'semantics', 'to_nix', 'plot'
        }
        
        if name in local_attributes:
            return object.__getattribute__(self, name)
        
        if 'repr' in self.__dict__:
            repr_obj = object.__getattribute__(self, 'repr')
            if hasattr(repr_obj, name):
                return getattr(repr_obj, name)

        return object.__getattribute__(self, name)

        
    def __str__(self):
        return self.repr.__str__()

    def plot(self, *, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        times = self.repr.times.rescale(pq.s).magnitude
        signal = self.repr._params["signal"]
        ax.plot(times, signal, **kwargs)

        ax.set_xlabel("Time")
        ax.set_ylabel("Rate")
        title = f"Rate ({self.method})"
        if self.kernel is not None:
            title += f" - {self.kernel}"
        ax.set_title(title)
        return ax

    def to_nix(self, nix_file_or_path, **kwargs):
        if not isinstance(nix_file_or_path, str):
            nix_file_or_path = str(nix_file_or_path)
        return self.repr.to_nix(nix_file_or_path=nix_file_or_path, **kwargs)
