"""
Contains 1-halo and 2-halo term P(k) functions.
"""

import numpy as np
from scipy.integrate import simpson

class PkBuilder:
    def __init__(self, cosmology, hod_model, prof_model):
        self.cosmo = cosmology
        self.hod = hod_model
        self.u = prof_model.u
        self.k = cosmology.k
        self.z = cosmology.z
        self.log10Mh = cosmology.log10Mh
        self.hmf = cosmology.hmf_grid
        self.hmfxbias = self.hmf * prof_model.bias
        
        self._cache_galaxy_integral()

    def _cache_galaxy_integral(self):
        """
        Cache I_gal(k,z) = ∫ dlogM HMF(M,z) * bias(M,z) * [Ncen(M,z) + Nsat(M,z) * u(k,M,z)]
        
        Galaxy term from A11 of 2204.05299
        """
        
        ncen = self.hod.ncen(self.Mh, self.z)  # shape (Nm, Nz)
        nsat = self.hod.nsat(self.Mh, self.z)  # shape (Nm, Nz)

        integrand = self.hmfxbias * (ncen + nsat * self.u)  # shape (Nm, Nz)
        
        Ig = simpson(integrand,x=self.log10Mh)
        
        # A12 of 2204.05299
        nbar = simpson(self.hmf * (ncen + nsat), x=self.log10Mh)
        
        self.Igal = Ig/nbar

    def compute_P_gg_2h(self):
        """
        Compute P_gg^2h(k,z) = P_lin(k,z) * I_gal^2
        """
        return self.cosmo.P_lin * self.I_gal**2
