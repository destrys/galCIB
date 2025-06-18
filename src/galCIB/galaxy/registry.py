# hod/registry.py

# Helps with user-defined HOD models 

from .hodmodel import HODModel

_HOD_MODEL_REGISTRY = {}

def register_hod_model(name, ncen_fn, nsat_fn):
    """
    Registers a new HOD model in the global registry.
    
    Parameters
    ----------
    name : str
        Unique name of the model.
    ncen_fn : callable
        Function for central galaxy occupation.
    nsat_fn : callable
        Function for satellite galaxy occupation.
    """
    if name in _HOD_MODEL_REGISTRY:
        raise ValueError(f"HOD model '{name}' is already registered.")
    _HOD_MODEL_REGISTRY[name] = (ncen_fn, nsat_fn)

def get_hod_model(name, z):
    """
    Retrieves a registered HOD model by name.
    
    Parameters
    ----------
    name : str
        Name of the model to retrieve.
    
    Returns
    -------
    HODModel
    """

    if name not in _HOD_MODEL_REGISTRY:
        raise KeyError(f"HOD model '{name}' not found.")
    ncen_fn, nsat_fn = _HOD_MODEL_REGISTRY[name]
    return HODModel(name, ncen_fn, nsat_fn, z)

def list_hod_models():
    """
    Returns the list of registered HOD model names.
    """
    return list(_HOD_MODEL_REGISTRY.keys())
