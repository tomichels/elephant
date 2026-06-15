import numpy as np
import quantities as pq
from .representation import Representation
import inspect
import functools

def store_params(func):
    """Decorator function to store params of function into class properties

    Args:
        func Callable: decorated function

    Returns:
        Callable: Wrapper function
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)

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


class MatrixObject(pq.Quantity, Representation):
    def __new__(
        cls,
        matrix,
        mtype=None,
        *,
        units=None,
        nix_obj=None,
        **kwargs,
    ):

        mtype = set(mtype) if mtype is not None else set()
        
        if isinstance(matrix, pq.Quantity):
            obj = matrix.view(cls)
        else:
            obj = pq.Quantity(matrix, units=units or pq.dimensionless).view(cls)
            
        obj.mtype = mtype

        Representation.__init__(obj, nix_obj=nix_obj)
        return obj
    
    @store_params
    def __init__(self, matrix, mtype=None, *, units=pq.dimensionless, nix_obj=None):
        self._validate_mtype()
        Representation.__init__(self, nix_obj=nix_obj)

    @property
    def mtype(self):
        return self._mtype

    
    def _validate_mtype(self):
        for condition in self.mtype:
            if condition == "symmetric" and not self.is_symmetric():
                raise ValueError("Matrix marked as symmetric but is not symmetric.")
            if condition == "square" and not self.is_square():
                raise ValueError("Matrix is not square but was defined as one")
            elif condition == "diagonal" and not self.is_diagonal():
                raise ValueError("Matrix is not diagonal but was defined as one")
    
    @mtype.setter
    def mtype(self, value):
        value = set(value)
        self._mtype = value
        self._validate_mtype()
            
    def is_square(self):
        rows, cols = self.shape
        return rows == cols
    
    def is_symmetric(self):
        if self.ndim != 2:
            return False
        mag = np.asarray(self.magnitude)
        return np.allclose(mag, mag.T, equal_nan=True)
    
    def is_diagonal(self):
        if self.ndim != 2:
            raise ValueError("Matrix is not 2D")
        mag = np.asarray(self.magnitude)
        return np.allclose(mag, np.diag(np.diag(mag)))
    
    def sanity_check(self):
        return f"matrix is symmetric: {self.is_symmetric()}, is square: {self.is_square()} is diagonal: {self.is_diagonal()}"
    
    
    def to_nix(
        self,
        nix_file_or_path,
        *,
        block_name: str | None = None,
        da_name: str | None = None,
        overwrite: bool = False,
    ):
        """
        NIX export for MatrixObject.

        Args:
            nix_file_or_path: Open nixio.File or path to write to.
            block_name: Block to write into; created if absent.
            da_name: DataArray name; defaults to class name.
            overwrite: Replace existing DataArray with the same name.

        Returns:
            nixio.DataArray
        """
        import nixio as nix

        owned = not isinstance(nix_file_or_path, nix.File)
        nixfile = nix.File.open(nix_file_or_path, mode="w") if owned else nix_file_or_path

        if block_name in [b.name for b in nixfile.blocks]:
            block = nixfile.blocks[block_name]
        else:
            block = nixfile.create_block(block_name or self.__class__.__name__, "matrix")

        da_name = da_name or self.__class__.__name__

        if da_name in [da.name for da in block.data_arrays]:
            if not overwrite:
                raise RuntimeError(
                    f"DataArray '{da_name}' already exists in block '{block_name}'. "
                    "Set overwrite=True to replace it."
                )
            del block.data_arrays[da_name]

        mag = np.asarray(self.magnitude)
        data_array = block.create_data_array(
            da_name,
            "elephant.matrix",
            dtype=mag.dtype,
            data=mag,
        )
        data_array.unit = str(self.units.dimensionality.string)

        for _ in range(self.ndim):
            data_array.append_set_dimension()

        sec = nixfile.create_section(f"{da_name}_meta", "metadata")
        sec["mtype"] = str(sorted(self.mtype))
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

    def plot(self, *, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(2, 1)
        else:
            ax = np.asarray(ax).ravel()
            if len(ax) < 2:
                raise ValueError("plot() needs 2 axes; pass ax=None to create them automatically.")

        mag = np.asarray(self.magnitude)
        im = ax[0].imshow(mag, **kwargs)
        ax[0].set_title("MatrixObject (symmetric)" if "symmetric" in self.mtype else "MatrixObject")
        ax[0].set_xlabel("Index")
        ax[0].set_ylabel("Index")
        plt.colorbar(im, ax=ax[0])
        ax[1].plot(mag)
        plt.tight_layout()
        return ax