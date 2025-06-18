import numpy as np

def compute_BAR_grid(M, z, cosmo):
    """
    Compute the Baryon Accretion Rate (BAR) grid.
    """
    M = np.atleast_1d(M)
    z = np.atleast_1d(z)

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
