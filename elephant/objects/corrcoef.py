import quantities as pq

from elephant.objects.base import _MatrixAnalysisObject
from elephant.objects.matrix import MatrixObject


class CorrCoefObject(_MatrixAnalysisObject):
    """AnalysisObject returned by correlation_coefficient().

    Wraps a MatrixObject repr containing the NxN Pearson correlation matrix.
    Attribute access is proxied to the underlying MatrixObject, so indexing,
    shape, units, mtype, etc. all work transparently.
    """

    _own_attrs = frozenset({'binary', 'bin_size'})

    def __init__(
        self,
        repr: MatrixObject,
        *,
        binary: bool = False,
        bin_size: pq.Quantity | None = None,
    ):
        if not isinstance(repr, MatrixObject):
            raise TypeError(f"Expected MatrixObject, got {type(repr)}")
        super().__init__(repr)
        self.binary = binary
        self.bin_size = bin_size

    @property
    def semantics(self):
        n = self.repr.shape[0]
        desc = f"Pearson correlation coefficient ({n}x{n})"
        if self.bin_size is not None:
            desc += f", bin_size={self.bin_size}"
        if self.binary:
            desc += ", binary"
        return desc


class CovarianceObject(_MatrixAnalysisObject):
    """AnalysisObject returned by covariance().

    Wraps a MatrixObject repr containing the NxN covariance matrix.
    Attribute access is proxied to the underlying MatrixObject.
    """

    _own_attrs = frozenset({'binary', 'bin_size'})

    def __init__(
        self,
        repr: MatrixObject,
        *,
        binary: bool = False,
        bin_size: pq.Quantity | None = None,
    ):
        if not isinstance(repr, MatrixObject):
            raise TypeError(f"Expected MatrixObject, got {type(repr)}")
        super().__init__(repr)
        self.binary = binary
        self.bin_size = bin_size

    @property
    def semantics(self):
        n = self.repr.shape[0]
        desc = f"Covariance matrix ({n}x{n})"
        if self.bin_size is not None:
            desc += f", bin_size={self.bin_size}"
        if self.binary:
            desc += ", binary"
        return desc
