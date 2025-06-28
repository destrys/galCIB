# cib/registry.py

_SFR_MODEL_REGISTRY = {}
_SNU_MODEL_REGISTRY = {}

def register_sfr_model(name, sfr_fn):
    if name in _SFR_MODEL_REGISTRY:
        raise ValueError(f"SFR model '{name}' is already registered.")
    _SFR_MODEL_REGISTRY[name] = sfr_fn

def register_snu_model(name, snu_fn):
    if name in _SNU_MODEL_REGISTRY:
        raise ValueError(f"Sν model '{name}' is already registered.")
    _SNU_MODEL_REGISTRY[name] = snu_fn

def get_sfr_model(name):
    _lazy_register_defaults()
    if name not in _SFR_MODEL_REGISTRY:
        raise KeyError(f"SFR model '{name}' not found.")
    return _SFR_MODEL_REGISTRY[name]

def get_snu_model(name):
    _lazy_register_defaults()
    if name not in _SNU_MODEL_REGISTRY:
        raise KeyError(f"Sν model '{name}' not found.")
    return _SNU_MODEL_REGISTRY[name]

# Lazy loading defaults
_default_models_loaded = False
def _lazy_register_defaults():
    global _default_models_loaded
    if not _default_models_loaded:
        from . import default_sfr, default_snu
        default_sfr.register_default_sfr_models()
        default_snu.register_default_snu_models()
        _default_models_loaded = True