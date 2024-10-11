"""
This module contains functions important for halo model calculations.
"""

import numpy as np 
import scipy.special as ss
from scipy.integrate import simpson
import consts

# import halo consts
Mh = consts.Mh
kk = consts.Plin['kh'] #FIXME: make sure what the unit of k is, if k/h or k
power = consts.Plin['pk'] #FIXME: make sure what the unit of Pk is

# DONE
def uprof_mixed(prof_params, rho_crit, rad, dlnpk_dlnk):
    """
    Returns Fourier Transform of halo profile mimicing DESI ELGs. 
    
    According to 2306.06319, the ELG halo profile is:
    p(r|M) = f_exp * exp(-r/(r_s * tau)) * (1-fexp)[NFW](r|M).
    
    Args: 
        prof_params: profile parameters passed to function
            fexp : frac. of ELG satellites following exp. profile
            rs : 
            tau : exponential index
    """
    
    fexp, tau, lambda_NFW = prof_params
    
    nfw_term, rs_original = nfwfourier_u(lambda_NFW, rho_crit, 
                                         rad, dlnpk_dlnk)
    exp_term = expfourier_u(tau=tau, rs = rs_original)
    
    res = fexp * exp_term + (1-fexp) * nfw_term
    
    return res

def expfourier_u(tau, rs):
    """
    Returns Fourier Transform of the exp. decay profile. 
    
    Args:
        tau : index of the exponential slope
        lambda_NFW : extending the NFW profile cut-off factor
    Returns:
        res : (k, Mh, z)
    """
    
    rs_times_tau = rs*tau # (Mh, z)
    rs_times_tau = np.expand_dims(rs_times_tau, axis = 0) # (k, Mh, z)
    num = 2 * (rs_times_tau)**3
    denom = ((rs_times_tau)**2 * kk[:, np.newaxis, np.newaxis]**2 + 1)**2
    res = num/denom
    
    return res
    
def nfwfourier_u(lambda_NFW, rho_crit, rad, dlnpk_dlnk):
    """
    Returns Fourier Transform of the NFW profile.
    
    From 2.14 of 2310.10848.
    Form is: [ln(1+c) - c/(1+c)]^(-1) [cos(q)(Ci(q + cq) - Ci(q)) + sin(q)(Si(q + cq) - Si(q)) - sin(cq)/(1 + cq)]
    
    Note rs is rescaled as rs/lambda_NFW according to pg 24 of 2306.06319.
    This applies for ELG-type haloes. 
    
    Args:
        lambda_NFW : rescaling factor of rs
    """
    
    rs_original = r_star(rho_crit, rad, dlnpk_dlnk) # (Mh, z)
    rs_rescaled = rs_original/lambda_NFW
    c = nu_to_c200c(rad, dlnpk_dlnk) # (Mh, z)
    c_term = ampl_nfw(c) # (Mh, z)
    q = kk[:, np.newaxis, np.newaxis] * rs_rescaled[np.newaxis,:,:] # (k, Mh, z)
    
    # broadcast to match q
    c = np.expand_dims(c, axis = 0) 
    c_term = np.expand_dims(c_term, axis = 0) 
    
    Si_qcq, Ci_qcq = sine_cosine_int(q + c*q) # (k, Mh, z)
    Si_q, Ci_q = sine_cosine_int(q) # (k, Mh, z)
    
    cos_q_term = np.cos(q) * (Ci_qcq - Ci_q)
    sin_q_term = np.sin(q) * (Si_qcq - Si_q)
    sin_qc_term = np.sin(c * q)/(1 + c*q)
    
    unfw = c_term *(cos_q_term + sin_q_term - sin_qc_term) # (k, Mh, z)
    
    return unfw, rs_original

def ampl_nfw(c):
    """
    Dimensionless amplitude of the NFW profile.
    Gives:
        $\frac{1}{\log(1+c) - \frac{c}{1+c}}$
    """
    return 1. / (np.log(1.+c) - c/(1.+c))

def sine_cosine_int(x):
    """
    sine and cosine integrals required to calculate the Fourier transform
    of the NFW profile.
    $ si(x) = \int_0^x \frac{\sin(t)}{t} dt \\
    ci(x) = - \int_x^\infty \frac{\cos(t)}{t}dt$
    """
    
    si, ci = ss.sici(x)
    return si, ci

def r_star(rho_crit, rad, dlnpk_dlnk):
    """
    Characteristic radius also called r_s in other literature.
    Physically refers to the transition point where the halo
    goes from a more concentrated inner region to a more 
    diffuse outer region. 
    
    $ c \equiv \frac{r_{200}{r_s}$
    
    Returns:
        res : (Mh, z)
    """
    
    c_200c = nu_to_c200c(rad, dlnpk_dlnk) # (Mh, z)
    r200 = r_delta(delta_h=200, rho_crit=rho_crit) # (Mh, z)
    res = r200/c_200c  
    
    return res

#FIXME: citation for this? 
def nu_to_c200c(rad, dlnpk_dlnk):  # length of mass array
    
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
    _nu = nu_delta(rad) # shape (Mh, z)
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

# DONE
def nu_delta(rad):  # peak heights
    
    """
    Returns the size of the peak heights.
    
    We use r_delta rather than the simple Lagrangian radius.
    This will be used in c-M relation to calculate the NFW profile.
    
    Args:
        rad: #FIXME: what is it?
    """
    
    delta_c = 1.686  # critical density of the universe. Redshift evolution is small and neglected #FIXME: what is this?
    sig = sigma(rad) # shape (Mh, z)
    return delta_c / sig 

# DONE
def sigma(rad):
    
    """
    Returns matter variance for given power spectrum, 
    wavenumbers k and radius of the halo. 
    Radius is calculated below from mass of the halo. 
    It has to be noted that there's no factor
    of \Delta = 200 to calculate the radius.
    
    Args:
        rad: radius of variance size #FIXME: Unit?
    Returns:
        res: Variance at size of radius rad shape (Mh, z)
    """
    
    rk = rad[:,:,np.newaxis] * kk[np.newaxis,np.newaxis,:] # shape (Mh, z, kk)
    rest = power * kk**3 # shape (z, kk)
    
    lnk = np.log(kk) # shape (kk,)
    integ = rest*W(rk)**2 # shape (Mh, z, kk)
    
    sigm = (0.5/np.pi**2) * simpson(integ, x=lnk, axis=-1) # integrating along kk
    res = np.sqrt(sigm)

    return res

# DONE
def W(rk):
    """
    Returns Fourier Transform of top-hat window function.
    
    Limit of 1.4e-6 put as it doesn't add much to the final answer and
    helps for faster convergence.
    """
    
    greater_term = (3*(np.sin(rk) - rk * np.cos(rk)) / rk**3)
    res = np.where(rk > 1.4e-6, greater_term, 1)
    
    return res

#DONE
def r_delta(delta_h = 200, rho_crit = consts.rho_crit_ELG): #FIXME: how many z bins are we calculating rho_crit in
    """
    Returns radius of the halo containing amount of matter
    corresponding to delta times the critical density of 
    the universe inside that halo. 
    
    Args:
        delta_h : How many times the critical density to consider to define halo. Default 200.
        rho_crit : Cosmic critical density in units of Msol/Mpc^3, shape (z,)
    Returns:
        res : shape (Mh, z) in units of Mpc^3
    """

    r3 = 3*consts.Mh[:,np.newaxis]/(4*np.pi*delta_h*rho_crit[np.newaxis,:]) ##FIXME: define delta_h
    
    res = r3**(1./3.)
    return res

#FIXME: do we infer for c? 
def ampl_nfw(c):
    """
    Returns dimensionless amplitude of the NFW profile.
    
    Equation from 2.14 of 2310.10848.
    Form: [ln(1 + c) - c/(1+c)]^(-1)
    
    Args:
        c : concentration parameter
    Returns:
        a : amplitude value
    """
    
    a = 1. /(np.log(1.+c) - c/(1.+c))
    return a

def get_dlnpk_dlnk():
    """
    When the power spectrum is obtained from CAMB, slope of the ps wrt k
    shows wiggles at lower k which corresponds to the BAO features. Also
    at high k, there's a small dip in the slope of the ps which is due to
    the effect of the baryons (this is not very important for the current
    calculations though). We are using the analysis from the paper
    https://arxiv.org/pdf/1407.4730.pdf where they have used the power
    spectrum from Eisenstein and Hu 1998 formalism where these effects have
    been negelected and therefore they don't have the wiggles and the bump.
    In order to acheive this, we have to smooth out the
    slope of the ps at lower k. But we have checked that the results
    do not vary significantly with the bump.
    """
    
    grad = np.zeros_like(power) # shape (z, kk)
    grad[:,:-1] = np.diff(np.log(power)) / np.diff(np.log(kk))
    
    #FIXME: is this not just copying the last value again? 
    grad[:,-1] = (np.log(power[:,-1]) - np.log(power[:,-2]))/(np.log(kk[-1]) - np.log(kk[-2]))
    kr = consts.k_R # (Mh,)
    
    res = np.zeros((len(kr), power.shape[0])) # shape(Mh, z)
    
    for i in range(power.shape[0]):
        res[:,i] = np.interp(kr, kk, grad[i]) # loop over per redshift
    
    return res