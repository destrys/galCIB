"""
Helper function and constants for analysis
"""

from astropy.cosmology import Planck18 as planck
from astropy import constants as apconst
from scipy import constants as spconst
import numpy as np
import pandas as pd 
import pickle

# global variables 
speed_of_light = spconst.c # in SI units
k_B = spconst.k # Boltzmann constant in SI units
hp = spconst.h # Planck's constant in SI units

KC = 1.0e-10  # Kennicutt constant for Chabrier IMF in units of Msol * yr^-1 * Lsol^-1
L_sun = 3.828e26 # From Abhishek's code 
fsub = 0.134 # from Abhi's code, further note below.
#         fraction of the mass of the halo that is in form of
#         sub-halos. We have to take this into account while calculating the
#         star formation rate of the central halos. It should be calulated by
#         accounting for this fraction of the subhalo mass in the halo mass
#         central halo mass in this case is (1-f_sub)*mh where mh is the total
#         mass of the halo.
#         for a given halo mass, f_sub is calculated by taking the first moment
#         of the sub-halo mf and and integrating it over all the subhalo masses
#         and dividing it by the total halo mass.


OmegaM0 = planck.Om0
Ode0 = planck.Ode0
H0 = planck.H0

# linear power spectrum
with open('data/plin_unit_h.p', 'rb') as handle:
    Plin = pickle.load(handle)

# halo mass function from colossus
with open('data/hmfz.p', 'rb') as handle:
    hmfz_dict = pickle.load(handle)
Mh = hmfz_dict['Mh']
hmfz = hmfz_dict['hmfz']

# store all relevant galaxy information
dict_gal = {}

# dict with ELG properties based on Karim et al. 2024
dict_gal['ELG'] = {}

dndz = pd.read_csv("data/gal/elg_fuji_pz_single_tomo.csv")
dict_gal['ELG']['z'] = dndz['Redshift_mid'].values
dict_gal['ELG']['pz'] = dndz['pz'].values
dict_gal['ELG']['mag_bias_alpha'] = 2.225

# cosmology-related values
dict_gal['ELG']['chi'] = planck.comoving_distance(dict_gal['ELG']['z'])
dict_gal['ELG']['Hz'] = planck.H(dict_gal['ELG']['z'])

def dchi_dz(z):
    """
    Returns differential element dchi/dz (z) 
    """
    
    a = apconst.c/(H0*np.sqrt(OmegaM0*(1.+z)**3 + planck.Ode0))
    return a.value
dict_gal['ELG']['dchi_dz'] = dchi_dz(dict_gal['ELG']['z'])

# dict with ELG HOD properties based on 
# Table 7 of Rocher et al. 2023
dict_gal['ELG']['HOD'] = {}
dict_gal['ELG']['HOD']['Ac'] = 0.1
dict_gal['ELG']['HOD']['log10Mc'] = 11.64
dict_gal['ELG']['HOD']['sigmaM'] = 0.30
dict_gal['ELG']['HOD']['gamma'] = 5.47
dict_gal['ELG']['HOD']['As'] = 0.41
dict_gal['ELG']['HOD']['alpha'] = 0.81
dict_gal['ELG']['HOD']['M0'] = 10**11.20
# based on Eq 3.9 of Rocher et al. 2023
dict_gal['ELG']['HOD']['M1'] = 10**13.84 * dict_gal['ELG']['HOD']['As']**(1/dict_gal['ELG']['HOD']['alpha'])

Omegab_to_OmegaM_over_z = planck.Ob(dict_gal['ELG']['z'])/planck.Om(dict_gal['ELG']['z'])
dict_gal['ELG']['Omegab_to_OmegaM_over_z'] = Omegab_to_OmegaM_over_z

# star formation constants
def BAR(M, z):
    """
    Returns baryon accretion rate for models M21 and Y23.
    
    Represents of the total amount of mass growth,
    what fraction is baryon.
    
    Returns:
        bar : of shape (M, z)
    """
    
    def MGR(M, z):
        """
        Returns Mass Growth Rate. 
        
        From 2.37 of 2310.10848.
        """
        
        # Reshape M and z to enable broadcasting
        M = M[:, np.newaxis]  # Shape (len(M), 1)
        z = z[np.newaxis, :]  # Shape (1, len(z))
        
        res = 46.1 * (M/1e12)**1.1 * (1 + 1.11 * z) * np.sqrt(OmegaM0 *(1+z)**3 + Ode0)
        
        return res
    
    bar = MGR(M, z) * Omegab_to_OmegaM_over_z
    
    return bar

bar = BAR(Mh, dict_gal['ELG']['z'])
