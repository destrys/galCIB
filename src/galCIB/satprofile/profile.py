"""
Contains class that handles inverse Fourier Transform
of the satellite profile inside the halo.
"""

import numpy as np 
from colossus.lss import bias
import warnings

from .default_models import compute_unfw, compute_nfw_exp_mixed
from .utils import _compute_default_r_delta, _compute_default_concentration, _compute_r_star, _compute_nu

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
        self.Mh = cosmology.Mh               # (k,M,z)
        self.z = cosmology.z               # (k,M,z)
        self.k_grid = cosmology.k_grid            # (k,M,z)
        self.theta = theta                  # for future model extension
        self.dlnpk_dlnk = cosmology.dlnpk_dlnk # concentration calculation
        self.profile_type = profile_type
        
        if self.profile_type == "mixed" and self.theta is not None:
            warnings.warn(
                "For the default mixed profile, theta should be [f_exp, tau_exp, lambda_NFW].",
                category=UserWarning
            )

        #Compute or load concentration and rstar parameters
        if c is None or rs is None:
            self.c, self.rs, self.nu = self._cache_params()  # load relevant params 
        else:
            self.c = c
            self.rs = rs

        # initialize u profile
        self.u = self._compute_u_profile()
        
        # initialize halo bias 
        self._compute_halo_bias()
        
    def _cache_params(self):
        """
        Initializes concentration and rs parameters 
        """
        
        r200 = _compute_default_r_delta(self.Mh, self.cosmo)
        c = _compute_default_concentration(r200, self.cosmo)
        rs = _compute_r_star(r200,c)
        nu = _compute_nu(self.cosmo)
        
        return c, rs, nu
        
    def _compute_u_profile(self):
        if self.profile_type == 'nfw':
            u = compute_unfw(self.k_grid, self.c, self.rs, 
                                lambda_NFW=1)
        elif self.profile_type == 'mixed':
            u = compute_nfw_exp_mixed(self.k_grid, self.c, self.rs,
                                        self.theta)
            
        return u
                
    def update_theta(self, theta):
        """
        Update model parameters and recompute the profile u(k, M, z).
        Only needed for theta-dependent models (e.g., 'mixed').
        """
        self.theta = theta
        self.u = self._compute_u_profile()
        
    def _compute_halo_bias(self, model=None):
        """
        Returns halo bias using colossus; defaults to Tinker10
        """
        
        self.hbias = bias.haloBiasFromNu(self.nu, self.z, mdef='200m')
        
        
        