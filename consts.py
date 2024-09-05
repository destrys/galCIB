"""
Helper function and constants for analysis
"""

from astropy.cosmology import Planck18 as planck
from astropy import constants as const

planck_const_h = const.h
speed_of_light = const.c 
boltzmann_kb = const.k_B

##FIXME: define the redshift bins here to pre-calculate 
## chi, dchi_dz
dndz_gal = 
pz = dndz_gal['dndz']
zrange = dndz_gal['zrange']

P_mm = #FIXME: define P_mm from CAMB
chi = planck.comoving_distance(zrange)
dchi_dz = speed_of_light/planck.H(zrange)