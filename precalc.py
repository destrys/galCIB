"""
This script precalculates functions that do not need to be called everytime.
This is to speed up calculations.
"""

import numpy as np

# import local modules
import cib 
import gal
import halo as h
import consts

## Speed up NFW profile ##

# Radius of halo containing 200 * rho_crit mass
rad200 = h.r_delta(delta_h = 200) # shape (Mh, z) # units 

# slope of the power spectrum
dlnpk_dlnk = h.get_dlnpk_dlnk() # shape (z, Mh)

# concentration param
concentration = h.nu_to_c200c(rad200, dlnpk_dlnk)

# amplitude param
concentration_amp = h.ampl_nfw(concentration)


##--RADIAL KERNEL--##
z_all = consts.Plin['z']
w_cib = cib.get_W_cib(z_cib=z_all)
w_gal = gal.get_Wgal(dict_gal=consts.dict_gal['ELG'])
w_mu = gal.get_Wmu(dict_gal=consts.dict_gal['ELG'])
w_gal_tot = w_gal + w_mu
w_cibxgal = w_gal_tot * w_cib
w_cibxcib = w_cib**2
w_galxgal = w_gal_tot**2

##--BIAS MODEL--##

# peak height as a function of halo mass 
nu = h.nu() #(Mh, z)

halo_biases = np.zeros_like(nu) # shape (Mh,z)

for i in range(len(halo_biases)): # looping over mass
    halo_biases[i] = h.b_nu(nu=nu[i],
                            z = consts.Plin['z'])
    
hmfzT = consts.hmfz.T