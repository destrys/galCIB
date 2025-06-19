"""
Contains class that handles inverse Fourier Transform
of the satellite profile inside the halo.
"""

import numpy as np 
import warnings

from .default_models import compute_unfw, compute_nfw_exp_mixed
from .utils import _compute_default_r_delta, _compute_default_concentration, _compute_r_star

warnings.simplefilter("once")

class SatProfile:
    """
    Satellite galaxy profile model. Computes u(k, M, z).

    Args:
        cosmology: Cosmology object with .k, .Mh, .z, and .rho_m
        theta: Optional parameter vector if profile or concentration depends on it
        profile_model: either 'nfw' or a custom callable (k, M, z, c) -> u
    """
        
    def __init__(self, cosmology, theta=None,
                 c=None, rs=None,
                 profile_type='nfw'):
        self.cosmo = cosmology
        self.Mh = cosmology.Mh               # (Nm,)
        self.z = cosmology.z                # (Nz,)
        self.k = cosmology.k                # (Nk,)
        self.theta = theta                  # for future model extension
        self.dlnpk_dlnk = cosmology.dlnpk_dlnk # concentration calculation
        self.profile_type = profile_type
        
        if self.profile_type == "mixed" and self.theta is not None:
            warnings.warn(
                "For the default mixed profile, theta should be [f_exp, tau_exp, lambda_NFW].",
                category=UserWarning
            )

        # Compute or load concentration and rstar parameters
        if c is None or rs is None:
            self.c, self.rs = self._load_c_rs()  # fallback to defaults
        else:
            self.c = c
            self.rs = rs

        self.u = _compute_u_profile()
        
        def _load_c_rs(self):
            """
            Initializes concentration and rs parameters 
            """
            
            r200 = _compute_default_r_delta(self.Mh, self.cosmo)
            c = _compute_default_concentration(r200, self.dlnpk_dlnk)
            rs = _compute_r_star(r200,c)
            
            return c, rs
        
        def _compute_u_profile(self):
            if self.profile_type == 'unfw':
                u = compute_unfw(self.k, self.c, self.rs, 
                                    lambda_NFW=1)
                return u
            elif self.profile_type == 'mixed':
                u = compute_nfw_exp_mixed(self.k, self.c, self.rs,
                                          theta)
                
    def set_theta(self, theta):
        """
        Update model parameters and recompute the profile u(k, M, z).
        Only needed for theta-dependent models (e.g., 'mixed').
        """
        self.theta = theta
        self.u = self._compute_u_profile()