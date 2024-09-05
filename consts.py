"""
Helper function and constants for analysis
"""

from astropy.cosmology import Planck18 as planck
from astropy import constants as const
import pandas as pd 

# global variables 
planck_const_h = const.h
speed_of_light = const.c 
boltzmann_kb = const.k_B

OmegaM = planck.Om0
H0 = planck.H0

##FIXME: define the redshift bins here to pre-calculate 
## chi, dchi_dz


P_mm = #FIXME: define P_mm from CAMB
dchi_dz = speed_of_light/planck.H(zrange)

# store all relevant galaxy information
dict_gal = {}

# dict with ELG properties based on Karim et al. 2024
dict_gal['ELG'] = {}

dndz = pd.read_csv("data/gal/elg_fuji_pz_single_tomo.csv")
dict_gal['ELG']['z'] = dndz['zrange'].values
dict_gal['ELG']['pz'] = dndz['dndz'].values
dict_gal['ELG']['mag_bias_alpha'] = 2.225

# cosmology-related values
dict_gal['ELG']['chi'] = planck.comoving_distance(dict_gal['ELG']['z'])
dict_gal['ELG']['Hz'] = planck.H(dict_gal['ELG']['z'])
