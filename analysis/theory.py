"""
This script generates theory C_ells based on fiducial values.
It accounts for p(z) variation to generate a collection of C_ells.

The goal is to pass these C_ells to the img-sys module and calculate
realistic covariance matrix that accounts for imaging weights and
p(z) variations. 

Last Updated: April 24, 2025 (Fixed synactical issues)
"""

import numpy as np
import pickle

import sys
sys.path.append("/Users/tkarim/research/galCIB/")
import consts
import powerspectra as ps 

# speed up healpy using healpy weights
healpy_data_path = '../healpy-weights/'

###--- DEFINE PARAMETERS ---###

# define fiducial values for M21 model
# 1- to 2- halo smoothing transition parameter is set to 1; but Mead++20 finds this to be 0.7 at z~0
hmalphas = np.ones(10) 

# shotnoise defined over log10 for sampling efficiency; 
# define as log10(N_shot/Jy/sr) = theta_shot;
# use unWISE green best-fit value because it is closest to ELG p(z)
shotnoise_gCIB = np.array([-1.8, -1.71, -2.24])
shotnoise_gCIB = shotnoise_gCIB - 2 # convert Y23 unit of 1e-8 MJy/sr to Jy/sr

# order is 353x353, 353x545, 353x857, 545x545, 545x857, 857x857
# taken from table 6 of 1309.0382, Planck 2013 CIB guesses
shotnoise_CIBCIB = np.log10(np.array([225, 543, 913, 1454, 2655, 5628]))
shotnoise_all = np.concatenate((shotnoise_gCIB, shotnoise_CIBCIB))

# physical parameters of importance
gal_params = np.array([5.47, 11.64, 0.30, 0.1, # Ncen (4): gamma, log10Mc, sigmaM, Ac
                       0.41, 11.20, 10**13.84 * (0.41)**(1/0.81), 0.81]) # Nsat (4): As, M0, M1, alpha
prof_params = np.array([0.58, 6.14, 0.67]) # fexp, tau, lambda_NFW
cib_params = np.array([0.49, 11.52, -0.02, 2.74, 0.5, 2.15, 11.38, 0.4]) # SFR (6): etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc, Mmin_IR, IR_sigma_lnM

all_params = np.concatenate((hmalphas, shotnoise_all, gal_params, prof_params, cib_params))

###--- END OF PARAMETERS ---###

# read pz values
# read all the pz values 

dndz_all = pickle.load(open("/Users/tkarim/research/galCIB/data/gal/dndz_extended.p", "rb"))

# loop over all the 1000 p(z) realizations
NSIDE = 1024
LMAX = 3 * NSIDE - 1

# store theory curves
theory_c_ells_ensemble = np.zeros((1000,10,LMAX)) # nsims x # of 2pcf x ell range (0,3*NSIDE-1)

# dictionary to store realizations of pz 
dict_pz = {}
dict_pz['z'] = dndz_all['zrange']

mag_bias = consts.dict_gal['ELG']['mag_bias_alpha']

for i in range (1000):
    if (i%100 == 0):
        print(i)
    dict_pz['pz'] = dndz_all['dndz'][i]
    theory_c_ells_ensemble[i] = ps.c_all(all_params, 
                                         cib_model='M21',
                                         pz=dict_pz,
                                         mag_bias_alpha=mag_bias)

np.save("/Users/tkarim/research/galCIB/data/theory_pz_variations_20250424.npy",theory_c_ells_ensemble)