"""
This module contains relevant information about 
DESI Legacy Imaging Surveys galaxies.
"""

import numpy as np
import scipy.special as ss
from scipy.integrate import simpson

# import local modules
import consts
import halo 

# read in cosmological constants
OmegaM0 = consts.OmegaM0

# read in halo constants
Mh = consts.Mh_Msol
log10Mh = consts.log10Mh

# read in ELG constants
gal_type = 'ELG'
dict_gal = consts.dict_gal[gal_type]
pz = dict_gal['pz']
z = dict_gal['z']

Ac = dict_gal['HOD']['Ac']
log10Mc = dict_gal['HOD']['log10Mc']
sigmaM = dict_gal['HOD']['sigmaM']
gamma = dict_gal['HOD']['gamma']
As = dict_gal['HOD']['As']
M0 = dict_gal['HOD']['M0']
M1 = dict_gal['HOD']['M1']
alpha = dict_gal['HOD']['alpha']

# read in CIB galaxy constants
#IR_sigma_lnM = consts.dict_gal['IR']['HOD']['sigma_lnM']

def Ncen_GHOD(log10Mc, sigmaM, Ac):
    """
    Returns num. of central galaxies per halo, as a function
    of halo mass. 
    Based on Gaussian HOD Model (GHOD) Eq. 3.1 of 2306.06319
    
    Args:
        log10Mc : characteristic mass for a halo to host a central galaxy
        sigmaM : width of the distribution
        Ac : sets the size of the central galaxy sample
    Returns:
        res : A single number #FIXME: do we make log10Mc and sigmaM z dependent?
    """
    
    prefact = Ac/(np.sqrt(2 * np.pi) * sigmaM)
    exp_term = -0.5/(sigmaM**2) * (log10Mh - log10Mc)**2
    res = prefact * np.exp(exp_term)
    
    return res

def z_evolution_model(params):
    """
    Returns redshift evolution model of HOD mass params.
    """
    
    mu_0, mu_p = params
    mu_X = mu_0 + mu_p * z/(1+z)
    
    return mu_X
    
def Ncen(hod_params, gal_type):
    """
    Returns num. of central galaxies per halo, between 0 and 1
    as a function of halo mass. 
    
    IR is based on 2.11 of 2310.10848.
    ELG is based on High Mass Quenched Model (mHMQ) Eq. 3.4 of 2306.06319
    
    Args:
        hod_params : HOD parameters to be constrained
            ELG hod_params: (gamma, log10Mc, sigmaM)
        gal_type : whether galaxy is ELG or IR
    Returns:
        res : vector of size (Mh,)
    """
    
    if gal_type == 'ELG':
        # Functional form is:
        # <N_c(M)> = <N_c^(GHOD)(M)>[1+erf(gamma*(log10(Mh/Mc))/(sqrt(2)*sigmaM))]
        # From 3.4 of 2306.06319.
        
        gamma, log10Mc, sigmaM, Ac = hod_params
        erf_term = gamma * (log10Mh - log10Mc)/(np.sqrt(2) * sigmaM)
        second_term = (1 + ss.erf(erf_term))
        first_term = Ncen_GHOD(log10Mc, sigmaM, Ac)
        res = first_term * second_term
    
    elif gal_type == 'IR':
        Mmin, IR_sigma_lnM = hod_params
        #Mmin = z_evolution_model(hod_params) #FIXME: figure out if z evolving or not
        erf_term = np.log(Mh/Mmin)/IR_sigma_lnM
        res = 0.5 * (1 + ss.erf(erf_term)) 
        
    else:
        print("not ELG or IR galaxies.")

    return res
        
def Nsat(hod_params):
    """
    Returns num. of sat. gal. per halo
    
    Args:
        hod_params : HOD parameters satellite galaxies
            ELG:
                As : sets the size of the satellite galaxy sample
                M0 : cut-off halo mass from which satellites can be present
                M1 : Normalization factor
                alpha : controls the increase in satellite richness 
                        with increasing halo mass
    Returns:
        res : vector of size (Mh,)
    """
    
    As, M0, M1, alpha = hod_params
    
    # flag for halo masses for which Mh - M0 > 0;
    # if Mh - M0 <= 0, then Nsat = 0. #FIXME: logic?
    flag = (Mh - M0) > 0
    res = np.zeros_like(Mh)
    res[flag] = As * ((Mh[flag] - M0)/M1)**alpha
    
    return res

def get_Wmu(dict_gal = dict_gal):
    """
    Returns magnification bias kernel as a func. 
    of redshift of the galaxies.
    """
    
    z = dict_gal['z']
    pz = dict_gal['pz']
    chi = dict_gal['chi']
    Hz = dict_gal['Hz']
    mag_bias_alpha = dict_gal['mag_bias_alpha']
    
    
    mag_bias_prefact = 3 * OmegaM0/(2 * consts.speed_of_light)
    mag_bias_prefact = (mag_bias_prefact * consts.H0**2/Hz * (1 + z) * chi).decompose() # to reduce to the same unit
    
    integrated_values = np.zeros_like(z)
    for i in range(len(z)): # loop over to get integrand values 
      zspecific_indx = i
      
      # only consider bins above the specific index 
      flag = (z >= z[zspecific_indx])
      ratio = chi[zspecific_indx]/chi[flag]
      
      # assuming constant alpha over z 
      integrand_term = (1 - ratio) * (mag_bias_alpha - 1) * pz[flag]
      integrated_values[i] = simpson(y = integrand_term, x = z[flag])
  
    mag_bias_term = mag_bias_prefact * integrated_values

    return mag_bias_term.value # shape (z,)

def get_Wgal(dict_gal = dict_gal):
    """
    Returns galaxy radial kernel as a func
    of redshift.
    """
    
    pz = dict_gal['pz']
    
    return pz
    

def galterm(params, u, gal_type = 'ELG'): #FIXME: needs testing
    """
    Returns the second bracket in A13 of 2204.05299.
    Form is (Nc + Ns * u(k, Mh, z)).
    This corresponds to the galaxy term in calculating Pk.
    
    Note: For ELG we are only picking one effective redshift, hence
    no evolution of Ncen or Nsat. 
    
    Args:
        params: MCMC parameters to be passed to HOD and 1-halo
                radial profile. The order of parameters are:
                gamma, log10Mc, sigmaM, Ac, 
                As, M0, M1, alpha,
                fexp, tau, lambda_NFW. 
                    
    Returns:
        res : shape (k X Mh X z)
    """
    
    params_Nc = params[:4]
    params_Ns = params[4:8]
    
    Nc = Ncen(params_Nc, gal_type = gal_type) # (Mh,)
    Ns = Nsat(params_Ns) # (Mh,)
    #u = uprof_mixed(params_prof, rho_crit, rad, dlnpk_dlnk) # (k, Mh, z)
    
    res = Nc[np.newaxis, :] + Ns[np.newaxis, :] * u
    
    return res 