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



## pre-calculate radial kernels
#radial_window_gal = gal.window_gal()
#radial_window_cib = cib.window_cib(nu_list, z)

