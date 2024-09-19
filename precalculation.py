"""
This module precalculates important parameters to speed up analysis
"""

import numpy as np 
import gal_prop as gp
import consts 

# import constants 
Mh = consts.Mh

## pre-calculate central and satellite distributions 
Nsat_precalc = gp.Nsat(Mh)
Ncen_precalc = gp.Ncen(Mh)
