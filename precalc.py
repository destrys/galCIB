"""
This script precalculates functions that do not need to be called everytime.
This is to speed up calculations.
"""

import halo as h
import consts

# Radius of halo containing 200 * rho_crit mass
rad200 = h.r_delta(delta_h = 200, rho_crit=consts.rho_crit_ELG) # shape (Mh, z)

# slope of the power spectrum
dlnpk_dlnk = h.get_dlnpk_dlnk() # shape (z, Mh)
#nu_to_c200c_val = h.nu_to_c200c(rad, dlnpk_dlnk)
