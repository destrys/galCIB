"""
This module defines the fixed cosmology used in the analysis.
This is helpful to pass consistent values from and to CAMB and colossus.
"""

import pickle 
import numpy as np

from astropy.cosmology import Planck18 as planck

import camb
from camb import model as camb_model

from colossus.cosmology import cosmology as cc
from colossus.lss import mass_function

# Redshift range covering both ELG and CIB in bins of ELG
z_all = np.insert(np.arange(0.05, 10.22, 0.1), 0, 0.)

# k range covering up to k_max = 10
k_all = np.logspace(-4, 1, 500)

##--CAMB--##
pars = camb.set_params(H0=planck.H0.value, 
                       ombh2=planck.Ob0 * planck.h**2, 
                       omch2=planck.Odm0 * planck.h**2, 
                       mnu=planck.m_nu.value[-1], 
                       num_nu_massive = (planck.m_nu.value > 0).sum(),
                       omk=planck.Ok0, 
                       tau=0.0543, As=np.exp(3.0448)/10**10, ns=0.96605, #Plik best fit Planck 2018 Table 1 left-most col 1807.06209
                       halofit_version='mead', lmax=2000)

pars.set_matter_power(redshifts = z_all, kmax=10.0)

# Linear spectra
pars.NonLinear = camb_model.NonLinear_none
results = camb.get_results(pars)

# Linear P_mm interpolator for z_all and k_all grid
# in units of little h 
PK = results.get_matter_power_interpolator(nonlinear=False, 
                                           hubble_units=False, 
                                           k_hunit=False)
PKgrid = PK.P(z_all, k_all)

SAVE = True

if SAVE:
    # dictionary to store with relevant information 
    plin_dict = {}

    plin_dict['z'] = z_all
    plin_dict['k'] = k_all
    plin_dict['pk'] = PKgrid

    with open('data/plin_unit_Mpc.p', 'wb') as handle:
        pickle.dump(plin_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

##--colossus--##
colossus_planck_cosmo = cc.fromAstropy(astropy_cosmo = planck, 
                      sigma8 = 0.809, # based on CAMB using the As 
                      ns = 0.96605, 
                      cosmo_name = 'planck_baseline')

# set halo mass bins
# range from 10e7 to 10e15 to cover almost all CIB galaxies 
#FIXME: make sure that the range is 7,15 in units of NOT little h
#Mh_Msol_h = np.logspace(7, 15, 100) # in units of little h
Mh_Msol = np.logspace(7, 15, 100)
Mh_Msol_h = Mh_Msol * planck.h 

cc.setCurrent(colossus_planck_cosmo) # set cosmology defn

# NOTE: colossus uses unit Msol/h for mass
hmfz = np.zeros((len(z_all), len(Mh_Msol_h))) #HMF of shape (z, Mh)

# NOTE: convert dn/dlnM to dn/dlog_10M, using ln(10) factor. 
for i in range(len(z_all)):
    hmfz[i] = mass_function.massFunction(x = Mh_Msol_h,
                           z = z_all[i],
                           mdef = '200m',
                           model = 'tinker08',
                           q_in = 'M',
                           q_out = 'dndlnM') * np.log(10) * planck.h  #FIXME: in Abhi's code, Anthony multiply with h^3, also compare with 218 and 219 in hmf_unfw_bias.py

    
if SAVE:
    # save hmf dictionary information
    hmfz_dict = {}
    hmfz_dict['z'] = z_all
    hmfz_dict['M_Msol_h'] = Mh_Msol_h
    hmfz_dict['hmfz_log10M'] = hmfz
    
    with open('data/hmfz_h.p', 'wb') as handle:
        pickle.dump(hmfz_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
