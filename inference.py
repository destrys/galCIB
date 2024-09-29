"""
This module performs all the necessary
inference steps for parameter estimation.
"""

import numpy as np 
import pocomc as pc

# modules in this folder
import precalculation as pc
import powerspectra
import cib
import gal

params_cib = {}
params_hod = {}

#FIXME: read data and invcov

#FIXME: define ordering of theta
#FIXME: for pocomc prior has to be set up differently
def log_prior(theta):
    etamax, mu_p0, mu_pp, sigma_M0, tau, zc, delta = theta
    
    # priors from Table 3 in 2310.10848
    if ((etamax<0.01) | (etamax>10) | (mu_p0<10) | (mu_p0>14) | (mu_pp<-5) | (mu_pp>5) | (sigma_M0<0.01) | (sigma_M0>4.) | (tau<0) | (tau>1) | (zc<0.5) | (zc>3.) | (delta<2) | (delta>5)):
        return -np.inf
    else:
        return 0
    
def log_likelihood(theta):
    """
    Returns gaussian likelihood
    """
    
    model = powerspectra.cibgalcross_cell_tot(theta)
    log_ll = (model - data) @ invcov @ (model - data)
    
    return log_ll

def log_posterior(theta):
    if log_prior == -np.inf:
        return log_prior
    else:
        return log_prior(theta) + log_likelihood(theta)

# sampling step
ndim = len(theta)

# Initialise sampler
sampler = pc.Sampler(
    prior=prior,
    likelihood=log_likelihood,
    vectorize=True,
    random_state=0)

# Start sampling
sampler.run()

# Get the results
samples, weights, logl, logp = sampler.posterior()