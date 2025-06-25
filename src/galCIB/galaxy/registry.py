# hod/registry.py

# Helps with user-defined HOD models 

from .hodmodel import HODModel

_HOD_MODEL_REGISTRY = {}

def register_hod_model(
    name,
    ncen_fn,
    nsat_fn,
    use_log10M_ncen=True,
    use_log10M_nsat=True,
    uses_z_ncen=True,
    uses_z_nsat=True,
):
    """
    Register a new HOD model config in the global registry.

    Note: cosmo is NOT passed here, user supplies it at get_hod_model().
    """
    if name in _HOD_MODEL_REGISTRY:
        raise ValueError(f"HOD model '{name}' is already registered.")
    
    _HOD_MODEL_REGISTRY[name] = {
        "ncen_fn": ncen_fn,
        "nsat_fn": nsat_fn,
        "use_log10M_ncen": use_log10M_ncen,
        "use_log10M_nsat": use_log10M_nsat,
        "uses_z_ncen": uses_z_ncen,
        "uses_z_nsat": uses_z_nsat,
    }

def get_hod_model(name, cosmo):
    """
    Retrieve a registered HOD model by name, instantiate with user-supplied cosmo.
    """
    if name not in _HOD_MODEL_REGISTRY:
        raise KeyError(f"HOD model '{name}' not found. Available: {list(_HOD_MODEL_REGISTRY.keys())}")
    
    config = _HOD_MODEL_REGISTRY[name]
    
    return HODModel(
        name=name,
        ncen_fn=config["ncen_fn"],
        nsat_fn=config["nsat_fn"],
        use_log10M_ncen=config["use_log10M_ncen"],
        use_log10M_nsat=config["use_log10M_nsat"],
        uses_z_ncen=config["uses_z_ncen"],
        uses_z_nsat=config["uses_z_nsat"],
        cosmo=cosmo,
    )


def list_hod_models():
    return list(_HOD_MODEL_REGISTRY.keys())

def is_model_registered(name):
    return name in _HOD_MODEL_REGISTRY