"""
Helper function and constants for analysis
"""

from astropy.cosmology import Planck18 as planck
from astropy import constants as const
import pandas as pd 
import pickle 

# global variables 
planck_const_h = const.h
speed_of_light = const.c 
boltzmann_kb = const.k_B

OmegaM = planck.Om0
H0 = planck.H0

# linear power spectrum
with open('data/plin_unit_h.p', 'rb') as handle:
    Plin = pickle.load(handle)

# halo mass function 
with open('data/hmfz.p', 'rb') as handle:
    hmfz_dict = pickle.load(handle)
Mh = hmfz_dict['Mh']
hmfz = hmfz_dict['hmfz']

# ##FIXME: define the redshift bins here to pre-calculate 
# ## chi, dchi_dz
# dchi_dz = speed_of_light/planck.H(zrange)

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