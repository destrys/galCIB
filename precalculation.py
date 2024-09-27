"""
This module precalculates important parameters to speed up analysis
"""

import numpy as np 
import consts 
import cib

# import constants 
Mh = consts.Mh
hmfz = consts.hmfz

# import planck cosmology
planck = consts.planck

# import galaxy information
dict_gal = consts.dict_gal['ELG']
z = dict_gal['z']

## pre-calculate necessary variables from cib.py
Td = cib.Tdust(z) # dust temperature in gal. bins

# Planck frequencies for CIB are: (100, 143, 217, 353, 545, 857) and IRAS (3000) GHz frequencies
nu_list = np.array([100, 143, 217, 353, 545, 857, 3000]) * 1e9   # convert GHz to Hz 
Bnu = cib.B_nu(nu_list, Td)


# ## pre-calculate central and satellite distributions 
# Ncen = gp.Ncen(Mh)
# Nsat = gp.Nsat(Mh)

# # apply conformity bias? #FIXME: check if this is correct
# Nsat[Ncen <= 0] = 0

# nbar_gal = gp.nbar_gal(Ncen=Ncen, Nsat=Nsat, Mh = Mh,
#                        HMFz=hmfz)
# #galterm_pk = gp.galterm_Pk(Ncen, Nsat, unfw)

