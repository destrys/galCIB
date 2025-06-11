"""
Script repurposed from Abhishek Maniyar's DopplerCIB github. 

Author: Tanveer Karim
Last Updated: 12 Dec 2024 (Fixed CIB-CIB bugs; now matches DopplerCIB)
Updated: 18 Nov 2024 (Added CIB-CIB function)
Updated: 11 Mar 2025 (Added n(z) user-specified option)
"""

import numpy as np

# integrates using simpson method 
from scipy.integrate import simpson, trapezoid
#from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

# import local modules
import consts
import precalc as pc
import gal
import cib
import halo as h

# cosmology constants
Hz = consts.Hz_list
chi = consts.chi_list
geo = (consts.dchi_dz/consts.chi2).value # NOTE: power spectra is defined over the grid [0.05, 10.15] with spacing of delta_z = 0.1 

# halo constants 
Mh = consts.Mh_Msol
dm = consts.log10Mh[1] - consts.log10Mh[0]
hmfz = consts.hmfz
hmfzT = pc.hmfzT
biasmz = pc.halo_biases

hmfzTXbias = hmfzT * biasmz

# expand 
hmfzT = hmfzT[np.newaxis,:,:]
biasmz = biasmz[np.newaxis,np.newaxis,:,:]
Pk_lin = consts.Pk_array_over_ell[np.newaxis,:,:]

# precalc values
dlnpk_dlnk = pc.dlnpk_dlnk
rad200 = pc.rad200
concentration = pc.concentration
concentration_amp = pc.concentration_amp
wcib = pc.w_cib
wcibwgal = pc.w_cibxgal/(consts.dchi_dz).value # Eqn A17 of https://arxiv.org/pdf/2204.05299
wcibwcib = pc.w_cibxcib
wgalwgal = pc.w_galxgal/(consts.dchi_dz**2).value # Eqn A17 of https://arxiv.org/pdf/2204.05299
z_all = consts.Plin['z']
cc = consts.cc # color correction factor 

# geometric prefactor
prefact_gg = geo * wgalwgal
prefact_gcib = geo * wcibwgal
prefact_cibcib = geo * wcibwcib

# interpolate ell values    
ell_range = np.arange(consts.LMIN, consts.LMAX)
ell_sampled = consts.ell
ELL_sampled = np.logspace(np.log10(consts.LMIN), 
                          np.log10(consts.LMAX), 20)

# Precompute the correction factors for the upper triangle
def precompute_cc_correction(cc, num_channels=3):
    """
    Precompute the multiplicative correction factors for the unique pairs in the upper triangle.
    
    Args:
        cc: 1D array of size num_channels for color correction factors.
        num_channels: (int) number of CIB channels 
        
    Returns:
        ccXcc: 1D array of size num_channels*(num_channels + 1)/2 containing the correction factors.
    """
    idx_upper = np.triu_indices(num_channels)
    ccXcc = cc[idx_upper[0]] * cc[idx_upper[1]]  # Precompute pairwise corrections
    
    return ccXcc

ccXcc = precompute_cc_correction(cc) 

def bin_pcl(r=[],mat=[],r_bins=[]):
    """Sukhdeep's Code to bins data and covariance arrays

    Input:
    -----
        r  : array which will be used to bin data, e.g. ell values
        mat : array or matrix which will be binned, e.g. Cl values
        bins : array that defines the left edge of the bins,
               bins is the same unit as r

    Output:
    ------
        bin_center : array of mid-point of the bins, e.g. ELL values
        mat_int : binned array or matrix
    """

    bin_center=0.5*(r_bins[1:]+r_bins[:-1])
    n_bins=len(bin_center)
    ndim=len(mat.shape)
    mat_int=np.zeros([n_bins]*ndim,dtype='float64')
    norm_int=np.zeros([n_bins]*ndim,dtype='float64')
    bin_idx=np.digitize(r,r_bins)-1
    r2=np.sort(np.unique(np.append(r,r_bins))) #this takes care of problems around bin edges
    dr=np.gradient(r2)
    r2_idx=[i for i in np.arange(len(r2)) if r2[i] in r]
    dr=dr[r2_idx]
    r_dr=r*dr

    ls=['i','j','k','l']
    s1=ls[0]
    s2=ls[0]
    r_dr_m=r_dr
    for i in np.arange(ndim-1):
        s1=s2+','+ls[i+1]
        s2+=ls[i+1]
        r_dr_m=np.einsum(s1+'->'+s2,r_dr_m,r_dr)#works ok for 2-d case

    mat_r_dr=mat*r_dr_m
    for indxs in itertools.product(np.arange(min(bin_idx),n_bins),repeat=ndim):
        x={}#np.zeros_like(mat_r_dr,dtype='bool')
        norm_ijk=1
        mat_t=[]
        for nd in np.arange(ndim):
            slc = [slice(None)] * (ndim)
            #x[nd]=bin_idx==indxs[nd]
            slc[nd]=bin_idx==indxs[nd]
            if nd==0:
                mat_t=mat_r_dr[slc]
            else:
                mat_t=mat_t[slc]
            norm_ijk*=np.sum(r_dr[slc[nd]])
        if norm_ijk==0:
            continue
        mat_int[indxs]=np.sum(mat_t)/norm_ijk
        norm_int[indxs]=norm_ijk
    return bin_center, mat_int

def pcl_binned(theta, cib_model, M, 
               ell_range = ell_range,
               ELL_range = ELL_sampled):
    """
    Returns pseudo-C_ell.
    
    Args:
        theta : model parameters 
        M : coupling matrix (obtained from Skylens)
        B : binning operator (obtained from Skylens)
    Returns:
        pcl_combined : single vector [gg, gcib_357, gcib_545, gcib_857]
    """
    
    # calculate unbinned pcl
    c_all = c_all(theta, cib_model)
    
    # apply coupling matrix 
    pcl_gg = c_all[0] @ M['gg']
    pcl_gcib = c_all[1:7] @ M['gcib']
    pcl_cibcib = c_all[7:] @ M['cibcib']
    
    # apply binning
    pcl_gg_binned = bin_pcl(mat=pcl_gg,r=ell_range,r_bins=ELL_range)
    pcl_gcib_binned = bin_pcl(mat=pcl_gcib,r=ell_range,r_bins=ELL_range)
    pcl_cibcib_binned = bin_pcl(mat=pcl_cibcib,r=ell_range,r_bins=ELL_range)
    
    pcl_combined = np.concatenate(pcl_gg_binned, pcl_gcib_binned, pcl_cibcib_binned)
    return pcl_combined

def c_all(theta, cib_model, num_channels=3,
          pz=None, mag_bias_alpha=None, gal_type = 'ELG'):
    """
    Returns C_gg, C_gCIB, C_CIBCIB.
    
    Args:
        theta : list of parameters 
        cib_model : 'M21' or 'Y23'
        num_channels : (int) number of CIB channels
        pz : (array) redshift distribution in probability density form 
        mag_bias_alpha : (float) galaxy mag bias
    Returns:
        c_all_combined : vectors of 10 power spectra vectors
                        [C_gg,
                         C_gx353, C_gx545, C_gx857,
                         C_353x353, C_545x545, C_857x857,
                         C_353x545, C_353x857, C_545x857,
                        ]
    """
    
    ## parameters 
    
    # gg, gx{CIB}, {CIB_nu X CIB_nu'}; this value is 1 + 3 + 6 = 10 for the default 3 channels
    num_of_unique_Cls = 1+num_channels+(num_channels*(num_channels+1))//2
    hmalpha = theta[:num_of_unique_Cls] 
    
    hmalpha_gg = hmalpha[0] # pass to galcrossgal
    hmalpha_gcib = hmalpha[1:num_channels+1] # pass to galcrosscib; 1:4 for default 3 channels
    hmalpha_gcib = hmalpha_gcib[:,np.newaxis,np.newaxis]
    hmalpha_cibcib = hmalpha[num_channels+1:] # pass to cibcrosscib; 4: for default 3 channels
    hmalpha_cibcib = hmalpha_cibcib[:,np.newaxis,np.newaxis]
    
    # gx{CIB}, {CIB_low X CIB_high}; 9 values for default 3 channels
    shotnoise = theta[num_of_unique_Cls:2*num_of_unique_Cls-1] 
    shotnoise = 10**shotnoise # convert log-value to proper value
    shotnoise_gcib = shotnoise[:num_channels]
    shotnoise_gcib = shotnoise_gcib 
    shotnoise_cibcib = shotnoise[num_channels:]

    start_of_physical_params = 2*num_of_unique_Cls-1
    gal_params = theta[start_of_physical_params:start_of_physical_params+8] # Ncen (4): gamma, log10Mc, sigmaM, Ac
                           # Nsat (4): As, M0, M1, alpha
    prof_params = theta[start_of_physical_params+8:start_of_physical_params+8+3] # fexp, tau, lambda_NFW
    cib_params = theta[start_of_physical_params+8+3:] # SFR (6): etamax (only for M23) or L0 (only for Y23), mu_peak0, mu_peakp, sigma_M0, tau, zc
                       # SED (3): beta, T0, alpha (only for Y23)

    # uprof
    uprof = h.uprof_mixed(prof_params, rad200, concentration, 
                         concentration_amp) # (k, Mh, z)
    
    # galterm, cibterm, nbar_halo
    galterm, Nc, Nsat = gal.galterm(gal_params, uprof, 
                                    gal_type = gal_type)
    cibterm, djc, djsub = cib.cibterm(cib_params, uprof, cib_model)
    nbar_halo = gal.nbargal_halo(Nc, Nsat, hmfzT)
    c_gg = galcrossgal_cell_tot(hmalpha_gg, galterm, nbar_halo, Nc,
                                pz=pz, mag_bias_alpha=mag_bias_alpha)[0]
    c_gcib = cibcrossgal_cell_tot(hmalpha_gcib, galterm, cibterm, shotnoise_gcib,
                                  pz=pz,mag_bias_alpha=mag_bias_alpha)
    c_cibcib = cibcrosscib_cell_tot(hmalpha_cibcib, cibterm,
                                    djc, djsub, uprof, shotnoise_cibcib,nnu=num_channels)

    # color correction
    c_gcib = c_gcib * cc[:,np.newaxis]
    c_cibcib = ccXcc[:, np.newaxis] * c_cibcib
    
    # combine all
    c_all_combined = np.vstack((c_gg, c_gcib, c_cibcib))

    # interpolate to ell_lmax = NSIDE with delta ell = 1
    spl = CubicSpline(ell_sampled,c_all_combined,axis=1)
    c_all_combined = spl(ell_range)

    return c_all_combined

def cibcrossgal_cell_tot(hmalpha_gcib, galterm, cibterm, shotnoise,
                         pz = None,mag_bias_alpha=None): 
    """
    Returns C_{g, CIB} accounting for all halo terms.
    """
    
    # calculate Pk of both halo terms
    oneh = cibgalcross_pk_1h(galterm, cibterm)
    twoh = cibgalcross_pk_2h(galterm, cibterm)
    
    pk_oneh_plus_2h = (oneh**hmalpha_gcib + twoh**hmalpha_gcib)**(1/hmalpha_gcib)
    
    if pz is None: # decide whether to recalculate prefactors
        local_prefact_gcib = prefact_gcib 
    else:
        pz = interpolate_user_pz(pz)
        wgal = gal.get_Wgal(pz)
        wmu = gal.get_Wmu(pz,mag_bias_alpha=mag_bias_alpha)
        wgal_tot = (wgal + wmu)/(consts.dchi_dz).value ##FIXME: check if wmu should be divided by dchi/dz
        wcibwgal = wgal_tot * wcib
        local_prefact_gcib = geo * wcibwgal
    
    integrand = local_prefact_gcib * pk_oneh_plus_2h
    
    c_ell_1h_plus_2h = simpson(integrand, x = z_all, axis=2)
    c_ell_1h_plus_2h = c_ell_1h_plus_2h #* 1e-6 # convert from Jy to mJy
    tot = c_ell_1h_plus_2h + shotnoise[:,np.newaxis] # (nu, ell)
    
    return tot

def cibgalcross_pk_1h(galterm, cibterm):
        """
        Returns 1-halo term of CIB X g power spectrum.
        
        From A13 of 2204.05299.
        Form is: 
            Pk_1h = int HMF * g-term * CIB-term * dlogMh
            
        Args:
            galterm : Nc + Ns * unfw (nu, k, Mh, z)
            cibterm : jc + js * unfw (nu, k, Mh, z)
        Returns:
            pk_1h : (nu, k, z)
        """
        
        integrand = hmfzT * galterm * cibterm 
        pk_1h = simpson(y=integrand, dx=dm, axis = 2)

        return pk_1h 

def cibgalcross_pk_2h(galterm, cibterm, plot = False):
        """
        Check the calculation first using two different integrals for both CIB
        and galaxies, then check with a single integral for both after
        multiplying. If both are same, then we can save one integral.
        
        Pk_2h (nu, k, z) = b_g * b_CIB * P_lin/(nbar * jbar)
        
        b_g = int HMF(Mh, z) * b(Mh, z) * gal_term (k, Mh, z) dlogMh
        b_CIB = int HMF(Mh, z) * b(Mh, z) * cib_term (nu, k, Mh, z) dlogMh
        
        Returns:
            pk_2h : (nu, k, z)
        """
        # pk = self.uni.Pk_array(self.ell, self.z)
        
        # galaxy bias term: int HMF(Mh, z) * b(Mh, z) * gal_term (k, Mh, z) dlogMh
        
        integrand = hmfzTXbias * galterm # (k,Mh,z)
        integral_g = simpson(y=integrand, dx=dm, axis=1) #(k,z)
        
        # CIB bias term: int HMF(Mh, z) * b(Mh, z) * cib_term (nu, k, Mh, z) dlogMh
        integrand = hmfzTXbias * cibterm
        integral_cib = simpson(y=integrand, dx=dm, axis=2) # (nu,k,z)
        pk_2h = integral_g[np.newaxis,:,:] * integral_cib * Pk_lin
        
        if plot:
            return pk_2h, integral_g, integral_cib
        else:
            return pk_2h

###--C_gal,gal---###

def galcrossgal_cell_tot(hmalpha_gg, galterm, nbar_halo, 
                         Nc, pz=None, mag_bias_alpha=None):
    """
    Returns C_gg total based on Pk.
    NOTE: it does not contain gg shot noise.
    
    Args:
        pz : binned galaxy redshift distribution density function 
    """

    p2h = galcrossgal_pk_2h(galterm, nbar_halo)
    p1h = galcrossgal_pk_1h(galterm, nbar_halo, Nc)
    
    p_tot = (p2h**hmalpha_gg + p1h**hmalpha_gg)**(1/hmalpha_gg)
    
    if pz is None:
        local_prefact_gg = prefact_gg
    else:
        pz = interpolate_user_pz(pz)
        wgal = gal.get_Wgal(dict_gal=pz)
        wmu = gal.get_Wmu(pz,mag_bias_alpha=mag_bias_alpha)
        wgal_tot = (wgal + wmu)/(consts.dchi_dz).value ##FIXME: check if wmu should be divided by dchi/dz
        wgalwgal_tot = wgal_tot**2
        local_prefact_gg = geo * wgalwgal_tot
    
    integrand = local_prefact_gg * p_tot
        
    c_ell = simpson(integrand, x=z_all, axis=-1)
    
    return c_ell
    
def interpolate_user_pz(pz):
    """
    Interpolate user-defined pz on the same grid as Pk.
    
    Args: 
        pz: dict with keywords 'z' and 'pz' corresponds to the redshift and the density values.
    """    
    
    pz_interpd = {}
    pz_interpd['pz'] = np.interp(z_all, pz['z'], pz['pz'], left = 0, right = 0)
    
    return pz_interpd
    
    
    
def galcrossgal_pk_2h(galterm, nbar_halo):
    """
    Returns the 2-halo term of the 3D Pk. 
    
    Based on A11 of 2204.05299.
    
    P2h = Plin * integrand^2
    integrand = HMF * (Nc+Ns*u) * b * dlogMh
    """
    
    integrand = (galterm * hmfzTXbias[np.newaxis,:,:])/nbar_halo
    integral = simpson(integrand,dx=dm,axis=1)
    p2h = Pk_lin * integral**2
    
    return p2h
    
def galcrossgal_pk_1h(galterm, nbar_halo, ncen):
    """
    Returns the 2-halo term of the 3D Pk. 
    
    Based on A10 of 2204.05299.
    
    P1h = int HMF * (2*Nc*Ns*u + Ns^2u^2) dlogMh
    
    Note integrand can be written as: 
    (Nc + Ns*u)^2 - Nc^2.
    = galterm^2 - Nc^2 Saves calculation time
    
    Args:
        ncen : num. of centrals as a function of Mh (Mh,)
        galterm : (Nc + Ns*u) (k,Mh,z)
    """    
    
    Nc = ncen[np.newaxis,:,np.newaxis]
    #nbar_halo = gal.nbargal_halo(ncen, nsat, hmfzT[0])
    
    integrand = galterm**2 - Nc**2
    integrand = integrand * hmfzT/nbar_halo**2
    
    p1h = simpson(integrand,dx=dm,axis=1)
    
    return p1h
    
def galcrossgal_cell_2h(galterm, nbar_halo):
    """
    Returns the 2-halo term of C_gg.
    """
    
    p2h = galcrossgal_pk_2h(galterm, nbar_halo)[0]
    integrand = geo * wgalwgal * p2h
    
    c_2h = simpson(integrand,x=consts.Plin['z'],axis=-1)
    
    return c_2h
    
def galcrossgal_cell_1h(ncen, nsat, galterm):
    """
    Returns the 1-halo term of C_gg
    """
    
    p1h = galcrossgal_pk_1h(ncen,nsat,galterm)
    integrand = geo * wgalwgal * p1h
    
    c_1h = simpson(integrand,x=consts.Plin['z'],axis=-1)
    
    return c_1h, geo * wgalwgal
    
###--C_CIB,CIB--###
def cibcrosscib_cell_tot(hmalpha_cibcib, cibterm, 
                         djc, djsub, uprof, 
                         shotnoise, nnu=3):
    """
    Returns the total CIB X CIB for all the nus.
    
    Args:
        hmalpha_cibcib : 3D power spectra relaxation parameter, 
            (nu, nu', 1, 1) where the last two axes correspond to k and z.
            Note that the matrix must be symmetric because Pk is invariant under
            nu <-> nu' transformation.
        cibterm : emissivity term (djc + djsub * uprof) (nu,Mh,z)
        djc : central emissivity term (nu,Mh,z)
        djsub: satellite emissivity term (nu,Mh,z)
        uprof : Fourier transform of the sat. radial prof (k,Mh,z)
        shotnoise : Pedestal shotnoise value of CIB-emitting galaxies,
            (nu,nu',1) where last axis corresponds to ell. 
            Note that the matrix must be symmetric because Pk is invariant under
            nu <-> nu' transformation.
        nnu : (int) number of CIB frequencies    
        
    Returns:
        c_ell : of shape (nu, nu', ell) in UNITS OF mJy^2/sr
    """
    
    # store C_ell
    c_ell = np.zeros((nnu, nnu, len(consts.ell)))
                
    # calculate 3D power spectra
    p2h = cibcrosscib_pk_2h(cibterm)
    p1h = cibcrosscib_pk_1h(djc, djsub, uprof)
    
    # Get the indices for the upper triangular part (including the diagonal)
    triu_indices = np.triu_indices(nnu)

    # Extract the unique upper triangular elements along the first two axes
    p2h = p2h[triu_indices[0],triu_indices[1],:,:]
    p1h = p1h[triu_indices[0],triu_indices[1],:,:]
    
    ptot = (p1h**hmalpha_cibcib + p2h**hmalpha_cibcib)**(1/hmalpha_cibcib)
    
    # calculate C_ell
    integrand = prefact_cibcib * ptot 
    c_ell = simpson(integrand, x=z_all, axis=-1) # over z
    
    # add shot noise
    c_ell = c_ell + shotnoise[:,np.newaxis] 
    
    # convert from Jy^2/sr to mJy^2/sr
    #c_ell = c_ell * 1e-12 
    
    return c_ell
            
def cibcrosscib_pk_2h(cibterm):
    """
    Returns P_{CIB X CIB'} 2-halo term.
    
    Note that the jnu terms get cancelled out by the ones in W_CIB
    
    Cell = int dz/c * H(z)/chi^2(z) * W_nu * W_nu' * Plin * integral_nu * integral_nu'
    integral_nu = integral_nu' = int dlog10Mh * I-term * b(Mh, z) * HMF
    
    Args:
        cibterm: (nu,k,z)
    Returns: 
        P_CIBXCIB : of shape (k,z)
    """
    
    # integrals
    integrand = cibterm * hmfzTXbias 
    integral = simpson(y=integrand, dx=dm, axis=2) #(nu,k,z)
    
    # Pairwise products of integrals for unique combinations (nu_i, nu_j)
    pk2h_all = np.einsum('ikz,jkz->ijkz', integral, integral)  # Shape (nu, nu, k, z)
    pk2h_all = Pk_lin * pk2h_all 
    
    return pk2h_all
    
def cibcrosscib_pk_1h(djc, djsub, uprof):
    """
    Returns P_{CIB X CIB'} 1-halo term.
    
    Note that the jnu terms get cancelled out by the ones in W_CIB
    
    pk1h = int(t1+t2+t3)*HMF*dlogMh
    t1 = djc_nu * djsub_nu' * unfw
    t2 = djc_nu' * djsub_nu * unfw
    t3 = djsub_nu * djsub_nu' * unfw^2 
    
    Args:
        djc_nu, djc_nu_prime : central emissivity (Mh,z)
        djsub_nu, djsub_nu_prime : sat emissivity (Mh,z)
        uprof : Fourier halo profile (ell,Mh,z)
    Returns: 
        pk1h : of shape (k,z)
    """
    
    # extend dimensions to match with uprof
    
    djc_re = djc[:,np.newaxis,:,:]
    djsub_re = djsub[:,np.newaxis,:,:]
    
    pk1h = np.zeros((djc.shape[0],djc.shape[0],
                     uprof.shape[0], djc.shape[-1])) # (nu,nu,k,z)
    
    for nu1 in range(djc.shape[0]): # loop over nu
        for nu2 in range(djc.shape[0]): # loop over nu'
            t1 = djc_re[nu1]*djsub_re[nu2]*uprof
            t2 = djc_re[nu2]*djsub_re[nu1]*uprof
            t3 = djsub_re[nu1]*djsub_re[nu2]*uprof**2
            
            integrand = t1 + t2 + t3
            
            pk1h[nu1,nu2] = simpson(integrand * hmfzT, 
                                    dx=dm,axis=1)
        
    return pk1h