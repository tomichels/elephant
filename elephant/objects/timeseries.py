from .representation import Representation

import quantities as pq
import numpy as np
import nixio as nix
import neo
import functools
import inspect

def store_params(func):
    """Decorator function to store params of function into class properties

    Args:
        func Callable: decorated function

    Returns:
        Callable: Wrapper function
    """
    sig = inspect.signature(func)

    named_params = {
        name for name, p in sig.parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        storable_kwargs = {k: v for k, v in kwargs.items() if k in named_params}
        bound = sig.bind_partial(*args, **storable_kwargs)
        bound.apply_defaults()
        params = {
            k: v for k, v in bound.arguments.items()
            if sig.parameters[k].kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }

        if len(args) > 0 and hasattr(args[0], '__dict__'):
            obj = args[0]
            params.pop('self', None)
            if not hasattr(obj, '_params'):
                obj._params = {}
            obj._params.update(params)
        else:
            wrapper._params = params

        return func(*args, **kwargs)

    return wrapper



class TimeSeriesObject(neo.AnalogSignal, Representation):
    def __new__(
    cls,
    signal,
    sampling_period=None,
    sampling_rate=None,
    *,
    units,
    t_start=0 * pq.ms,
    t_stop=None,
    nix_obj=None,
    **kwargs,
    ):
        if sampling_period is None:
            if sampling_rate is None:
                raise ValueError("Either sampling_period or sampling_rate must be provided.")
            sampling_period = 1 / sampling_rate

        neo_obj = neo.AnalogSignal.__new__(
            cls,
            signal=signal,
            sampling_period=sampling_period,
            units=units,
            t_start=t_start,
            t_stop=t_stop,
        )
        Representation.__init__(neo_obj, nix_obj=nix_obj)
        return neo_obj
    
    @store_params
    def __init__(
        self,
        signal,
        sampling_period=None,
        sampling_rate=None,
        *,
        units,
        t_start=0 * pq.ms,
        t_stop=None,
        kernel=None,
        nix_obj=None,
        **kwargs,
    ):
        if sampling_period is None:
            if sampling_rate is None:
                raise ValueError("Either sampling_period or sampling_rate must be provided.")
            sampling_period = 1 / sampling_rate

        super().__init__(
            signal=signal,
            units=units,
            t_start=t_start,
            t_stop=t_stop,
            sampling_period=sampling_period,
            **kwargs,
        )

        Representation.__init__(self, nix_obj=nix_obj)


    def to_nix(
        self,
        nix_file_or_path,
        *,
        block_name: str | None = None,
        da_name: str | None = None,
        overwrite: bool = False,
    ):
        """
        NIX export for TimeSeriesObject.

        Args:
            nix_file_or_path: Open nixio.File or path to write to.
            block_name: Block to write into; created if absent.
            da_name: DataArray name; defaults to class name.
            overwrite: Replace existing DataArray with the same name.

        Returns:
            nixio.DataArray
        """
        owned = not isinstance(nix_file_or_path, nix.File)
        nixfile = nix.File.open(nix_file_or_path, mode="w") if owned else nix_file_or_path

        if block_name in [b.name for b in nixfile.blocks]:
            block = nixfile.blocks[block_name]
        else:
            block = nixfile.create_block(block_name or self.__class__.__name__, "recording")

        da_name = da_name or self.__class__.__name__

        if da_name in [da.name for da in block.data_arrays]:
            if not overwrite:
                raise RuntimeError(
                    f"DataArray '{da_name}' already exists in block '{block_name}'. "
                    "Set overwrite=True to replace it."
                )
            del block.data_arrays[da_name]

        data_array = block.create_data_array(
            da_name,
            "neo.analogsignal",
            dtype=np.asarray(self).dtype,
            data=self.magnitude,
        )
        data_array.unit = str(self.units.dimensionality.string)

        dim = data_array.append_sampled_dimension(
            float(self.sampling_period.rescale(self.sampling_period.units).magnitude)
        )
        dim.unit = str(self.sampling_period.units)
        dim.label = "time"
        dim.offset = float(self.t_start.rescale(self.sampling_period.units))

        sec = nixfile.create_section(f"{da_name}_meta", "metadata")
        sec["neo_type"] = "AnalogSignal"
        if hasattr(self, "_params"):
            for k, v in self._params.items():
                try:
                    sec[k] = str(v)
                except Exception:
                    pass
        data_array.metadata = sec

        self.nix_obj = data_array

        if owned:
            nixfile.close()
        return data_array

    @property
    def sampling_interval(self):
        return self.sampling_period