"""
This module precalculates important parameters to speed up analysis
"""

import numpy as np 

# import local modules
import consts 
import cib
import gal

# import constants 
Mh = consts.Mh
hmfz = consts.hmfz

# import planck cosmology
planck = consts.planck

# import galaxy information
dict_gal = consts.dict_gal['ELG']
z = dict_gal['z']

## pre-calculate necessary variables from cib.py
# Planck frequencies for CIB are: (100, 143, 217, 353, 545, 857) and IRAS (3000) GHz frequencies
nu_list = np.array([100, 143, 217, 353, 545, 857, 3000]) * 1e9   # convert GHz to Hz 

Td = cib.Tdust(z) # dust temperature in gal. bins
Bnu = cib.B_nu(nu_list, Td)
mod_Bnu = cib.mod_blackbody(Bnu, nu_list)
nu0_z = cib.nu0_z(Td)

# normalization factor for Theta
#tmpbnu = np.diag(cib.B_nu(tmpnu0, Td))

## pre-calculate radial kernels
radial_window_gal = gal.window_gal()
#radial_window_cib = cib.window_cib(nu_list, z)

## precalculation for SFR 
bar = cib.BAR(M=Mh, z=z)

