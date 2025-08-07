"""
This script generates theory C_ells based on fiducial values.
It accounts for p(z) variation to generate a collection of C_ells.

The goal is to pass these C_ells to the img-sys module and calculate
realistic covariance matrix that accounts for imaging weights and
p(z) variations. 

Last Updated: July 16, 2025: Updates based on the new package structure

Version History:
- April 24, 2025 (Fixed synactical issues)
"""

import numpy as np
import pickle
from galCIB import SnuModel, CIBModel, PkBuilder, AnalysisModel

# speed up healpy using healpy weights
healpy_data_path = '../healpy-weights/'

###--- DEFINE PARAMETERS ---###

#mag_alpha = 1 
mag_alpha = 2.225
NSIDE = 1024
LMAX = 3 * NSIDE - 1
NSIMS = 1000
ks = np.logspace(-3,1,500)
Mh = np.logspace(7,15,100)

# define fiducial values for M21 model
# 1- to 2- halo smoothing transition parameter is set to 1; but Mead++20 finds this to be 0.7 at z~0
theta_hmalpha = np.array([0.7]) # Use Mead+2020 value 

# shotnoise defined over log10 for sampling efficiency; 
# define as log10(N_shot/Jy/sr) = theta_shot;
# use unWISE green best-fit value because it is closest to ELG p(z)
shotnoise_gCIB = np.array([-1.8, -1.71, -2.24])
shotnoise_gCIB = 10**(shotnoise_gCIB - 2) # convert Y23 unit of 1e-8 MJy/sr to Jy/sr

# order is 353x353, 353x545, 353x857, 545x545, 545x857, 857x857
# taken from table 6 of 1309.0382, Planck 2013 CIB guesses
shotnoise_CIBCIB = np.log10(np.array([225, 543, 913, 1454, 2655, 5628]))
#shotnoise_all = np.concatenate((shotnoise_gCIB, shotnoise_CIBCIB))

# physical parameters of importance
theta_cen = np.array([3.28, 11.49, 0.45, 0.1]) # gamma, log10Mc, sigmaM, Ac
theta_sat = np.array([0.38, 10**11.14, 10**13., 0.59]) # As, M0, M1, alpha_sat
theta_prof = np.array([0.52, 9.07, 0.7]) # fexp, tau, lambda_NFW

theta_sfr_M21 = np.array([0.49, #eta_max 
                      11.52,-0.02, #mu0_peak, mup_peak
                      2.74,0.5,2.15]) #sigmaM0, tau, zc
theta_snu = np.array([2.7, 1.98, 21.13, 0.21, 1.7]) #L0, beta_dust, T0, alpha_dust, gamma_dust
theta_IR = np.array([11.38, 2.6, 0.4]) # mu0_Mmin, mup_Mmin, sigma_lnM

###--- END OF PARAMETERS ---###

# read pz values
# read all the pz values 
dndz_all = pickle.load(open("/Users/tkarim/research/galCIB/data/gal/dndz_extended.p", "rb"))
dndz = dndz_all['dndz']
zrange = dndz_all['zrange']

# store theory curves
#theory_c_ells_ensemble = np.zeros((NSIMS,10,LMAX)) # (Nsims, NCell, Nell)

## Setup CIB 
nu_obs = [353, 545, 857] # Planck effective freq. in GHz

# load Planck filter response curves 
from galCIB.utils.io import load_my_filters
fpath = "/Users/tkarim/research/galCIB"
cib_filters = load_my_filters(f"{fpath}/data/filters/",
                              nu_obs=nu_obs)

## Setup Survey 
from galCIB import Survey 
LMIN = 0
ells = np.arange(LMIN, LMAX)
nbins = 20
binned_ell_ledges = np.logspace(np.log10(LMIN), 
                                np.log10(LMAX),
                                nbins)

# Setup cosmology
from galCIB import Cosmology
cosmo = Cosmology(zrange, ks, Mh, 
                  colossus_cosmo_name='planck18',
                  use_little_h=False)

# Setup ELG HOD 
from galCIB import get_hod_model
elg_hod_model = get_hod_model("DESI-ELG", cosmo)

# Setup Satellite Profile Model 
from galCIB import SatProfile
elg_sat_profile = SatProfile(cosmo, theta_prof,
                             profile_type='mixed')

# Setup SFR Model
from galCIB import SFRModel
# IR-emitting galaxies  HOD model 
hod_IR = get_hod_model("Zheng05", cosmo)
sfr_model = SFRModel(name="M21", 
                     hod=hod_IR, 
                     fsub=0.134)

# Setup arrays to save values in 
cgg = np.zeros((NSIMS, LMAX))
cgI = np.zeros((NSIMS, 3, LMAX)) # 3 channels
cII = np.zeros((NSIMS, 6, LMAX)) # 6 channels 

for i in range (1000):
    
    if (i%100 == 0):
        print(i)
        
    # define Survey and vary dndz 
    elg_survey = Survey(z=zrange, 
                 pz=dndz[i],
                 mag_alpha=mag_alpha,  # galaxy-specific
                 cib_filters=cib_filters,  # dict: freq_GHz -> (freq_array_Hz, response_array)
                 ells=ells, nside=NSIDE, 
                 binned_ell_ledges=binned_ell_ledges,
                 name="ELG-Planck")
    
    # Compute window 
    elg_survey.compute_windows(cosmo,True)
    
    # Setup Snu Model 
    # Non-parametric Snu model (M21)
    snu_model_M21 = SnuModel(name="M21", 
                            cosmo=cosmo,
                            survey=elg_survey,
                            nu_prime=np.array([353, 
                                               545, 
                                               857]))
    
    # Initialize CIB model
    cib_M21 = CIBModel(sfr_model=sfr_model, 
                    snu_model=snu_model_M21,
                    hod_IR=hod_IR)
    
    # Setup Pk Object
    # M21 
    pk_survey_M21 = PkBuilder(hod_model=elg_hod_model,
                              cib_model=cib_M21,
                              prof_model=elg_sat_profile
                      )
    
    
    # Setup Analysis Model 
    analysis = AnalysisModel(survey=elg_survey,
                             pk3d=pk_survey_M21)
    
    # Compute C_ells
    cgg[i], tcgI, tcII = analysis.update_cl(theta_cen=theta_cen,
                        theta_sat=theta_sat,
                        theta_prof=theta_prof,
                        theta_sfr=theta_sfr_M21,
                        theta_snu=theta_snu,
                        theta_IR_hod=theta_IR,
                        #theta_sn_gI=shotnoise_gCIB,
                        #theta_sn_II=shotnoise_CIBCIB,
                        theta_sn_gI=np.zeros(3),
                        theta_sn_II=np.zeros(6),
                        hmalpha=theta_hmalpha)
    
    cgI[i] = tcgI#.flatten()
    cII[i] = tcII#.flatten()
    
np.savez("/Users/tkarim/research/galCIB/data/theory_pz_variations_20250722_noshot.npz",
        cgg=cgg, cgI=cgI, cII=cII)