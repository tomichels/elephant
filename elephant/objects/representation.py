from abc import ABC, abstractmethod


class Representation(ABC):
    def __init__(self, *, nix_obj=None):
        self.nix_obj = nix_obj

    @abstractmethod
    def to_nix(
        self, nix_file_or_path, *, block_name=None, da_name=None, overwrite=False
    ):
        raise NotImplementedError
