# src/galCIB/__init__.py

from .cosmology import Cosmology

# Remove all imports, leave empty or just __all__
# __all__ = [
#     "clustering",
#     "cosmology",
#     "consts",
#     "powerspectra",
#     "precalc",
# ]


# from .survey import Survey
# from .power import PowerSpectra

__all__ = ['Cosmology']#, 'Survey', 'PowerSpectra']
__version__ = "0.0.1"