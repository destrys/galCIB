# cib/default_models.py

"""
Contains the default SFR model based on 2006.16329
"""

import numpy as np

from galCIB.galaxy.utils import evolving_log_mass

def sfr_default_factory(eta_fn, BAR_grid, z_ratio):
    """
    Returns the default SFR model 
    """
    
    def sfr_model(M, z, theta_eta):
        eta_vals = eta_fn(M, z, z_ratio, theta_eta)
        return eta_vals * BAR_grid
    
    return sfr_model

def eta_fn(M, z, theta_eta, **kwargs):
    """
    Returns star formation efficiency parameter.
    
    From 2.38 of 2310.10848.
    eta (M, z) = eta_max * exp(-0.5((ln M - ln Mpeak(z))/sigmaM(z))^2)
    M = Mh and z = z so these are fixed. 
    sigmaM represents range of halo masses over which star formation is efficient. 
    """
    
    z_ratio = kwargs.get('z_ratio', None)
    eta_max, mu0_peak, mup_peak, sigmaM0, tau, zc = theta_eta
    
    # Mpeak evolving as a func. of z 
    Mpeak_z = 10**evolving_log_mass(mu0_peak, mup_peak, z_ratio)
    
    # 2.39 of 2310.10848
    sigmaM_z = np.where(M < Mpeak_z, sigmaM0, 
                      sigmaM0*(1-tau/zc * np.maximum(0, zc-z)))  
    
    expterm = -0.5 * ((np.log(M) - np.log(Mpeak_z))/sigmaM_z)**2
    
    eta_z = eta_max * np.exp(expterm)
    
    return eta_z
