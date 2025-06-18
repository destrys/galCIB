"""
Contains default callable HOD models. 
Includes DESI ELG mHMQ, GHOD and Zheng05 models.
"""

import numpy as np
from scipy.special import erf

from .utils import evolving_log_mass
from .registry import register_hod_model, get_hod_model

def Ncen_mHMQ(log10_Mh, theta, z=None):
    """
    Returns num. of central galaxies per halo, between 0 and 1
    as a function of halo mass. 
    
    ELG is based on High Mass Quenched Model (mHMQ) Eq. 3.4 of 2306.06319
    """
    
    gamma, log10Mc, sigmaM, Ac = theta
    
    erf_term = gamma * (log10_Mh - log10Mc)/(np.sqrt(2) * sigmaM)
    second_term = 1 + erf(erf_term)
    first_term = Ncen_GHOD(log10_Mh, (log10Mc, sigmaM, Ac))
    
    Ncen = first_term * second_term
    
    return Ncen

def Ncen_GHOD(log10_Mh, theta, z=None):
    """
    Returns num. of central galaxies per halo, as a function of halo mass. 
    Based on Gaussian HOD Model (GHOD) Eq. 3.1 of 2306.06319
    """
    
    log10Mc, sigmaM, Ac = theta
    
    exp_term = -0.5 ((log10_Mh - log10Mc)/sigmaM)**2
    prefact = Ac/(np.sqrt(2 * np.pi) * sigmaM)
    
    Ncen = prefact * np.exp(exp_term)
    
    return Ncen

def Ncen_Z05(Mh, theta, z=None, z_over_1plusz=None):
    """
    Returns num. of central. galaxies (0 or 1) per halo.
    Based on Eqn 2.11 from 2310.10848
        
    N_c(M) = 0.5 * (1 + erf (ln(M/M_min)/sigma_lnM))
    """
    
    mu0_Mmin, mup_Mmin, sigma_lnM = theta
    Mmin_z = 10**evolving_log_mass(mu0_Mmin, mup_Mmin, z_over_1plusz)
    
    erf_term = np.log(Mh/Mmin_z)/sigma_lnM
    
    Ncen = 0.5 * (1 + erf(erf_term))
    
    return Ncen
    
def Nsat_ELG(Mh, theta, z=None):
    """
    Returns num. of sat. gal. per halo.
    Based on 3.5 of 2306.06319.
    """
    
    As, M0, M1, alpha_sat = theta
    
    # if Mh - M0 < 0, then Nsat = 0
    Nsat = np.where(Mh-M0 < 0, 0, As * ((Mh-M0)/M1)**alpha_sat)
    
    return Nsat 

def Nsat_Z05(Mh, theta, z=None, z_over_1plusz=None, **kwargs):
    
    """
    Returns num. of sat gal. per halo. 
    Eqn 2.11 from 2310.10848
    
    Nsat(M) = Nc(M) * Heaviside(M-M0) * ((M - M0)/M1)**alpha_s
    """
    
    ncen = kwargs.get("ncen", 1.0) # this implementation expects ncen
    M0, mu0_M1, mup_M1, alpha_sat = theta
    Mh_M0 = Mh - M0
    M1_z = 10**evolving_log_mass(mu0_M1, mup_M1, z_over_1plusz)
    
    Nsat = np.heaviside(Mh_M0, 1)*(Mh_M0/M1_z)**alpha_sat
    
    if ncen is not None:
        Nsat *= ncen
    
    return Nsat 
        
# Register default models for access

register_hod_model("Zheng05", Ncen_Z05, Nsat_Z05)
register_hod_model("DESI-ELG", Ncen_mHMQ, Nsat_ELG)
register_hod_model("mHMQ", Ncen_mHMQ, Nsat_ELG)