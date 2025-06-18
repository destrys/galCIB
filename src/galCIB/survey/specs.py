"""
This module sets up the galaxy survey specifications used in
the analysis. 
"""

import numpy as np 

from .window import compute_Wcib, compute_Wg

class Survey:
    def __init__(self, z, pz, mag_alpha, name=None):
        self.z = z
        self.z_ratio = z/(1+z)
        self.pz = pz
        self.mag_alpha = self._standardize_mag_alpha(mag_alpha)
        self.name = name or "unnamed"

    def _standardize_mag_alpha(self, mag_alpha):
        """
        Standardizes mag_alpha, whether a scalar or a vector
        """
        if np.isscalar(mag_alpha):
            return np.full_like(self.z, mag_alpha)
        else:
            mag_alpha = np.asarray(mag_alpha)
            if mag_alpha.shape != self.z.shape:
                raise ValueError(f"mag_alpha shape {mag_alpha.shape} does not match z shape {self.z.shape}")
            return mag_alpha
    
    def compute_windows(self, cosmology):
        self.Wg = compute_Wcib(self.z)
        self.Wcib = compute_Wg(self.z, self.pz, use_mag_bias=True,
                                  mag_alpha=self.mag_alpha,
                                  cosmo=cosmology)
    