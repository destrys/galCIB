"""
This module contains relevant information about 
DESI Legacy Imaging Surveys galaxies.
"""

import numpy as np
import scipy.special as ss
from scipy.integrate import simpson
import consts

gal_type = 'ELG'
dict_gal = consts.dict_gal[gal_type]
pz = dict_gal['pz']

Ac = dict_gal['HOD']['Ac']
log10Mc = dict_gal['HOD']['log10Mc']
sigmaM = dict_gal['HOD']['sigmaM']
gamma = dict_gal['HOD']['gamma']
As = dict_gal['HOD']['As']
M0 = dict_gal['HOD']['M0']
M1 = dict_gal['HOD']['M1']
alpha = dict_gal['HOD']['alpha']

def Ncen_GHOD(Mh, Ac = Ac, sigmaM = sigmaM, log10Mc = log10Mc):
    """
    Returns num. of central galaxies per halo, as a function
    of halo mass. 
    Based on Gaussian HOD Model (GHOD) Eq. 3.1 of 2306.06319
    
    Args:
        Mh : halo mass 
        Ac : size of the central galaxy sample
        log10Mc : characteristic halo mass that hosts a central gal.
        sigmaM : width of distribution
    """
    
    prefact = Ac/(np.sqrt(2 * np.pi) * sigmaM)
    exp_term = -0.5/(sigmaM**2) * (np.log10(Mh) - log10Mc)**2
    exp_term = np.exp(exp_term)
    
    return prefact * exp_term

def Ncen(Mh, Ac = Ac, sigmaM = sigmaM, 
         log10Mc = log10Mc, 
         gamma = gamma, gal_type = 'ELG'):
    """Returns num. of central galaxies per halo, between 0 and 1
    as a function of halo mass. 
    Based on High Mass Quenched Model (mHMQ) Eq. 3.4 of 2306.06319
    
    Args:
        Mh : halo mass
        Ac : size of the central galaxy sample
        Mc : characteristic halo mass that hosts a central gal.
        sigmaM : width of distribution
        gamma : asymmetry of distribution 
        gal_type : galaxy type 
    """
    # m = self.mh
    if gal_type == 'ELG':
        
        erf_term = np.log10(Mh) - log10Mc
        erf_term *= gamma/(np.sqrt(2) * sigmaM)
        erf_term = ss.erf(erf_term)
        second_term = 1 + erf_term 
        first_term = Ncen_GHOD(Mh, Ac, sigmaM, log10Mc)
        
        return first_term * second_term
    else:
        print("not ELG")
        
def Nsat(Mh, As = As, M0 = M0,
         M1 = M1, alpha = alpha):
    """
    Returns num. of sat. gal. per halo
    
    Args:
        Mh : halo mass
        As : size of sat. gal. sample
        M0 : cut-off halo mass at which sat. gal. can be produced
        M1 : normalization constant 
        alpha : richness parameter
    """
    
    power_term = (Mh - M0)/M1
    return As * power_term**alpha

def window_gal():
    """
    Return radial kernel of galaxy sample.
    """
    
    return pz

def galterm_Pk(Ncen, Nsat, unfw): #FIXME: needs testing
    """
    Returns the second bracket in A13 of 2204.05299.
    This corresponds to the galaxy term in calculating Pk.
    
    Args:
        Ncen : num. of central galaxies inside a given halo (Mh X z)
        Nsat : num. of sat. galaxies inside a given halo (Mh X z)
        unfw : Fourier transform the NFW profile inside the halo. 
               Shape is num of modes X num of Halo mass func bins X num of redshifts
    
    Returns:
        res : shape (k X Mh X z)
    """
    
    assert Ncen.shape()[0] == 2, "Ncen must be of shape (Mh X z)"
    assert Nsat.shape()[0] == 2, "Nsat must be of shape (Mh X z)"
    assert unfw.shape()[0] == 3, "unfw profile must be of shape (k X Mh X z)"
    
    #reshape to include k dimension to multiply with unfw
    Nsat = Nsat[np.newaxis, :, :] 
    Ncen = Ncen[np.newaxis, :, :]
    res = Nsat * unfw 
    res = Ncen + res 
    
    return res 