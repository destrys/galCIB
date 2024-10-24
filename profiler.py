"""Profile the power spectra code."""
import numpy as np
import powerspectra as ps 

theta_M23 = np.array([1., # relaxtion parameter hmalpha
                0., # shotnoise
                5.47, 11.64, 0.30, 0.1, # Ncen (Y23): gamma, log10Mc, sigmaM, Ac
                0.41, 10**(11.20), 10**(13.78), 0.81, # Nsat (R23): As, M0, M1, alpha
                0.58, 6.14, 0.67, # radial profile (R23): fexp, tau, lambda_NFW
                0.49, 11.52, -0.02, 11.51, 0.55, 2.74, 0.5, 2.15, # SFR (M23): etamax, mu_peak0, mu_peakp, sigma_M0, tau, zc
                ])

tot  = ps.cibgalcross_cell_tot(theta_M23, cib_model='M21', plot=False)
