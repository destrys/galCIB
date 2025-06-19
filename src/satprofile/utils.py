"""
Utility file with commonly used functions.
"""

import numpy as np 
from scipy.integrate import simpson

def _compute_default_concentration(r200, dlnpk_dlnk):
    """
    Returns 
    
    Args:
    
    Returns:
        res : (Mh, z)
    """

    use_mean = False  # 2 relations provided. mean and median.
    phi0_median, phi0_mean = 6.58, 7.14
    phi1_median, phi1_mean = 1.37, 1.60
    eta0_median, eta0_mean = 6.82, 4.10
    eta1_median, eta1_mean = 1.42, 0.75
    alpha_median, alpha_mean = 1.12, 1.40
    beta_median, beta_mean = 1.69, 0.67
    _nu = _nu_delta(r200) # shape (Mh, z)
    n_k = dlnpk_dlnk # shape (Mh, z)
    
    if use_mean:
        c_min = phi0_mean + phi1_mean * n_k
        nu_min = eta0_mean + eta1_mean * n_k
        res = c_min * ((_nu/nu_min)**-alpha_mean + (_nu/nu_min)**beta_mean)/2
    else:
        c_min = phi0_median + phi1_median * n_k
        nu_min = eta0_median + eta1_median * n_k
        res = c_min * ((_nu/nu_min)**-alpha_median + (_nu/nu_min)**beta_median)/2
    
    return res

def _compute_default_r_delta(Mh, cosmology, delta_h=200):
    """
    Returns radius of the halo containing amount of matter
    corresponding to delta times the critical density of 
    the universe inside that halo. 
    
    Args:
        delta_h : How many times the critical density to consider to define halo. Default 200.
        rho_crit : Cosmic critical density in units of Msol/Mpc^3
    Returns:
        r : radius in units of Mpc 
    """

    r3 = 3*Mh/(4*np.pi*delta_h*cosmology.rho_crit) # [Mpc]^3
    r = r3**(1/3)

    return r

def _nu_delta(rad):
    """
    Returns the size of the peak heights given radius.
    
    We use r_delta rather than the simple Lagrangian radius.
    This will be used in c-M relation to calculate the NFW profile.
    """
    
    delta_c = 1.686  # critical density of the universe. Redshift evolution is small and neglected
    sig = _sigma(rad) # shape (Mh, z)
    return delta_c / sig 

def _sigma(rad, k, pk):
    
    """
    Returns matter variance for given power spectrum, 
    wavenumbers k and radius of the halo. 
    Radius is calculated below from mass of the halo. 
    It has to be noted that there's no factor
    of \Delta = 200 to calculate the radius.
    
    Args:
        rad: radius of variance size
    Returns:
        res: RMS at size of radius rad shape (Mh, z)
    """

    # need the full Pk to get the proper numerical integration 
    rk = np.outer(rad, k)
    rest = pk * k**3
    lnk = np.log(k)
    Wrk = _W(rk)
    
    integ = rest*Wrk**2

    sigm = (0.5/np.pi**2) * simpson(integ, x=lnk, axis=0) # integrating along kk
    res = np.sqrt(sigm)

    return res

def _W(rk): 
    """
    Returns Fourier transform of top hat window function.
    
    Limit of 1.4e-6 put as it doesn't add much to the final answer and
    helps for faster convergence
    """
    
    res = np.where(rk > 1.4e-6, (3*(np.sin(rk)-rk*np.cos(rk)) / rk**3), 1)
    
    return res 

def _compute_r_star(r200, c200c):
    """
    Characteristic radius also called r_s in other literature.
    Physically refers to the transition point where the halo
    goes from a more concentrated inner region to a more 
    diffuse outer region. 
    
    $ c \equiv \frac{r_{200}{r_s}$
    
    Returns:
        res : (Mh, z)
    """
    
    rstar = r200/c200c  
    
    return rstar
