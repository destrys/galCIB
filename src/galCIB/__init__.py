# src/galCIB/__init__.py

from . import clustering
from . import cosmology
from . import consts
from . import powerspectra
from . import precalc

# Direct functions exposed to users
from .powerspectra import c_all

__all__ = [
    "clustering",
    "cosmology",
    "consts",
    "powerspectra",
    "precalc",
    "c_all",
]