#survey/survey.py

"""
This module sets up the galaxy survey specifications used in
the analysis. 
"""

import numpy as np 
from scipy.integrate import simpson 
from .window import compute_Wcib, compute_Wg

class Survey:
    def __init__(self, z, 
                 pz=None, mag_alpha=None,  # galaxy-specific
                 cib_filters=None,  # dict: freq_GHz -> (freq_array_Hz, response_array)
                 nu_obs=None,   # optional override of effective frequencies
                 ells=None, nside=None,
                 name=None):
        
        """
        Container Class for Survey. 
        
        Args:
            z : redshift array
            pz : galaxy redshift distribution 
            mag_alpha : galaxy magnification bias parameter alpha 
            nu_obs : effective CIB frequencies
            filt_response : CIB filter response curves 
        """

        self.name = name or "unnamed"
        self.z = z
        self.z_ratio = z / (1 + z)
        
        # galaxy-specific
        self.pz = pz
        self.mag_alpha = self._standardize_mag_alpha(mag_alpha)
        
        # CIB-specific filters
        self.filters = cib_filters or {}
        
        # If user does not provide nu_obs explicitly, use keys of filters sorted
        if nu_obs is None and self.filters:
            self.nu_obs = sorted(self.filters.keys())
        else:
            self.nu_obs = nu_obs or []

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
    
    def compute_windows(self, cosmology, use_mag_bias=True):
        self.Wcib = compute_Wcib(self.z)
        self.Wg = compute_Wg(self.z, self.pz, use_mag_bias=use_mag_bias,
                                  mag_alpha=self.mag_alpha,
                                  cosmo=cosmology)
    
    def get_filter_response(self, freq_GHz):
        """
        Returns (freq_array_Hz, response_array) for given effective freq in GHz.
        """
        return self.filters.get(freq_GHz, (None, None))
    
    def apply_filter_to_sed(self, sed, freq_sed, filter_key):
        """
        Returns predicted flux of a given SED through a single filter.

        Args:
            sed : (Nz, Nwv) array of SEDs (Nz samples, Nwv frequencies)
            freq_sed : (Nwv,) array of frequencies for the SED
            filter_key : key to select filter from self.filters dict

        Returns:
            flux : (Nz,) flux for each SED through selected filter
        """
        filt_freq, filt_response = self.filters[filter_key]  # unpack filter arrays
        sed = np.atleast_2d(sed)  # ensure shape (Nz, Nwv)
        norm = simpson(filt_response, x=filt_freq)

        # Integrate each SED over the filter response with interpolation

        flux = np.array([simpson(np.interp(filt_freq, w_row, 
                                       s_row, left=0.0, right=0.0) * filt_response, 
                             x=filt_freq) for w_row, s_row in zip(freq_sed, sed)])

        return flux / norm
