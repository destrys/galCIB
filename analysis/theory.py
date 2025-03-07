"""
This script generates theory C_ells based on some fiducial values.
It accounts for p(z) variation to generate a collection of C_ells.

The goal is to pass these C_ells to the img-sys module and calculate
realistic covariance matrix that accounts for imaging weights and
p(z) variations. 

Last Updated: Dec 24, 2024
"""

import numpy as np

import sys
sys.path.append("/Users/tkarim/Documents/research/cib.nosync")

import powerspectra as ps 

C_CIBxCIB = ps.cibcrosscib_cell_



    hmalpha = theta[0]
    shotnoise = theta[1]
    gal_params = theta[2:10] # Ncen (4): gamma, log10Mc, sigmaM, Ac
                           # Nsat (4): As, M0, M1, alpha
    prof_params = theta[10:13] # fexp, tau, lambda_NFW
    cib_params = theta[13:] # SFR (6): etamax (only for M23) or L0 (only for Y23), mu_peak0, mu_peakp, sigma_M0, tau, zc
                       # SED (3): beta, T0, alpha (only for Y23)
    uprof = h.uprof_mixed(prof_params, 
                         rad200, 
                         concentration, 
                         concentration_amp) # (k, Mh, z)
    galterm = gal.galterm(gal_params, uprof, 
                          gal_type = 'ELG')
    cibterm = cib.cibterm(cib_params, uprof, cib_model)
    
    # radial kernal and prefactors 
    prefact = geo * wcibwgal
    
    # expand dims to pass 
    galterm = galterm[np.newaxis,:,:,:] #(nu,k,Mh,z)
    cibterm = cibterm[3:]
        
    # calculate Pk of both halo terms
    oneh = cibgalcross_pk_1h(galterm, cibterm)
    twoh = cibgalcross_pk_2h(galterm, cibterm)
    
    pk_oneh_plus_2h = prefact * (oneh**(1/hmalpha) + twoh**(1/hmalpha))**hmalpha
    pk_oneh_plus_2h[:,:,0] = 0 # z = 0 is 0, otherwise integral gets NaN
    c_ell_1h_plus_2h = simpson(pk_oneh_plus_2h, x = consts.Plin['z'], axis=2)
    tot = c_ell_1h_plus_2h+shotnoise
    
    if plot: # for plotting purposes
        return tot, prefact*oneh, prefact*twoh
    else:
        return tot 