#cib/utils.py

import numpy as np
from scipy.integrate import simpson

def SED_to_flux(sed, freq_sed, freq_filt, filt_response):
    """
    Returns the predicted flux per Planck channel of a given SED.
    
    Args:
        sed : (Nz, Nwv) array of SEDs
        wv_sed : (Nz, Nwv) array of wavelength grids for each SED
        fwv : (Nfwv,) common wavelength grid for filter response
        fresponse : (Nfwv,) filter response curve
    """
    # Ensure sed is 2D: (Nz, Nwv)
    sed = np.atleast_2d(sed)

    # Normalize the response curve
    norm = simpson(filt_response, x=freq_filt)

    # Interpolate and integrate in one list comprehension
    flux = np.array([simpson(np.interp(freq_filt, w_row, s_row, left=0.0, right=0.0) * filt_response, x=freq_filt) for w_row, s_row in zip(freq_sed, sed)])

    return flux / norm

def compute_BAR_grid(cosmo):
    """
    Compute the Baryon Accretion Rate (BAR) grid.
    """
    M = cosmo.Mh[:,np.newaxis] # (Nm,Nz)
    z = cosmo.z[np.newaxis,:] # (Nm,Nz)

    Om0 = cosmo.Om0 
    Ode0 = cosmo.Ode0

    Ob_z = cosmo.Ob_z
    Om_z = cosmo.Om_z
    
    # Mass Growth Rate (MGR) 
    # From 2.37 of 2310.10848 (originally 2 of 1001.2304).
    Msol = 1e12
    MGR = 46.1 * (M/Msol)**1.1 * (1 + 1.11*z) * np.sqrt(Om0*(1+z)**3 + Ode0)
    
    BAR = MGR * Ob_z/Om_z
    
    return BAR