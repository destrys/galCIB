"""
Contains radial window kernel calculating functions. 
"""

import numpy as np
from scipy.integrate import simpson

def compute_Wcib(z):
        """
        Returns radial kernel of CIB.
        
        Note that we do NOT scale the output by the 
        mean emissivity since the mean cancels out in
        C_ell calculation. For proper wcib, consider
        multiplying by mean emissivity. 
        """
        
        wcib = 1/(1 + z)
        
        return wcib
    
def _compute_Wmu(z, pz, mag_alpha, cosmo):
    """
    Returns radial kernel of magnification bias.
    
    Args:
        z : redshift range
        pz : redshift distribution
        mag_alpha : magnification bias alpha param. as a func. of z 
        cosmo: Cosmology object that helps with pre-factor
        calculations.
    """
    
    chi = cosmo.chi
    Hz = cosmo.Hz
    H0 = cosmo.H0
    Om0 = cosmo.Om0
    c = cosmo.c 
    
    # Eqn 6 of 1410.4502

    # prefactor = 3/2 * Omega_M0/c * H0^2/H(z) * (1+z) * chi(z)
    
    prefact = 3/2 * Om0/c * H0**2/Hz * (1+z) * chi
    
    # integral from z to zstar dz' * (1 - chi(z)/chi(z')) * (alpha(z')-1) * p(z')
    integral_term = np.zeros_like(z)
    for i in range(len(z)):
        chi_at_z = chi[i]
        mask_from_z_to_zstar = (z >= z[i])
        chi_from_z_to_zstar = chi[mask_from_z_to_zstar]
        alpha_from_z_to_zstar = mag_alpha[mask_from_z_to_zstar]
        pz_from_z_to_zstar = pz[mask_from_z_to_zstar]
        
        integrand = (1 - chi_at_z/chi_from_z_to_zstar) * (alpha_from_z_to_zstar - 1) * pz_from_z_to_zstar
        integral_term[i] = simpson(integrand,x=z[mask_from_z_to_zstar])
        
    wmu = prefact * integral_term
    
    return wmu
    
def compute_Wg(z, pz, use_mag_bias = True, mag_alpha = None, cosmo = None):
    """
    Returns radial kernel of galaxy survey.
    """
    
    wgal = pz * 1/cosmo.dchi_dz
    
    if use_mag_bias:
        if mag_alpha is None:
            raise ValueError("mag_alpha must be provided when use_mag_bias=True")
        wmu = _compute_Wmu(z,pz,mag_alpha,cosmo)
        print(f'wmu = {wmu}')
        wgal_tot = wgal + wmu
    else:
        wgal_tot = wgal
        
    return wgal_tot