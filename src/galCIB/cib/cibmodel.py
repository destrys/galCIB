"""
This sets up the CIB model class that can take user-defined SFR and Snu
models. It has the M21 and Y23 models implemented as defaults. 
"""

import numpy as np 
from .utils import compute_BAR_grid, SED_to_flux
from .registry import get_sfr_model, get_snu_model, _lazy_register_defaults


class CIBModel:
    """
    Container class for a CIB emissivity model.
    
    Attributes
    ----------
    name : str
        Unique identifier for the model.
    sfr_fn : callable
        Function that returns SFR(M, z) given M, z, and theta.
    snu_fn : callable
        Function that returns effective SED given M, z, and theta.
    cosmo : object 
        Providing cosmology-dependent quantities
    nu_prime : array
        If parametric S_nu model, what nu' = nu*(1+z) to sample SED over
    filtered : boolean
        True -> Use user-specified filters, False -> use unfiltered Snu
    """
    
    def __init__(self, sfr_fn=None, snu_fn=None, snu_model_name = "Y23",
                 hod=None, cosmo=None, nu_prime=None, 
                 filtered=True, filter_files=None, data_dir="../data/"):
        
        self.hod = hod
        self.cosmo = cosmo
        self.z_ratio = self.hod.z_ratio # z/(1+z)
        self.filtered = filtered
        self.filter_files = filter_files  # Optionally passed by user
        
        # Frequency grid setup
        self.nu_prime = nu_prime or self._generate_nu_prime_grid()
        
        # Precompute geometric prefactor, A6 and A7 of 2204.05299
        self.geom_prefact = cosmo.chi**2 * (1 + cosmo.z)
        KC = 1.0e-10  # Kennicutt constant for Chabrier IMF in units of Msol * yr^-1 * Lsol^-1
        self.geom_prefact_over_KC = self.geom_prefact/KC
        
        # Pre-register defaults once
        _lazy_register_defaults()

        # Setup SFR function
        if sfr_fn is None:
            self.BAR_grid = compute_BAR_grid(self.cosmo)
            sfr_factory = get_sfr_model("M21")
            self._sfr_fn = sfr_factory(self.BAR_grid, self.z_ratio)
        else:
            self._sfr_fn = sfr_fn
            
        # Setup Snu function and filtered flag
        if snu_fn is None or isinstance(snu_fn, str):
            # Determine the model name
            snu_model = snu_fn if isinstance(snu_fn, str) else snu_model_name
            
            snu_factory = get_snu_model(snu_model)
            
            if snu_model == "Y23":
                nu_prime_grid = self._generate_nu_prime_grid()
                self._snu_fn = snu_factory(nu_prime_grid, self.cosmo.z)
                self.filtered = True if filtered is None else filtered
                
            elif snu_model == "M21":
                if data_dir is None:
                    raise ValueError("Must specify `data_dir` for M21 SED model.")
                self._snu_fn = snu_factory(data_dir)
                self.filtered = False if filtered is None else filtered
                
            else:
                raise ValueError(f"Unknown snu model name '{snu_model}'")
                
        else:
            # snu_fn is a callable provided by user
            self._snu_fn = snu_fn
            self.filtered = False if filtered is None else filtered
        
    def compute_sfr(self, theta_sfr):
        return self._sfr_fn(self.cosmo.Mh_grid[0],
                            self.cosmo.z,
                            theta_sfr) # Pass Mh_grid[0] because of shape (Nm,Nz)
    
    def compute_snu(self, theta_snu):
        raw_sed = self._snu_fn(theta_snu)

        if self.filtered:
            if self.filters is None:
                raise RuntimeError("Filtered SED requested but no filters were loaded.")
            return self._apply_filters(raw_sed)
        else:
            return raw_sed

    def compute_djc(self, theta_sfr, theta_snu, theta_hod):
        
        """
        Returns the emissivity of central galaxies per log halo mass. 
        
        from A6 of 2204.05299
        djc_dlogMh (Mh, z) = chi^2 * (1+z) * SFRc/K * S^eff_nu (z)
        
        Args:
            theta_sfr : SFR parameters
            theta_snu : Snu parameters
        Returns:
            jc : matrix of shape (nu, Mh, z)
        """
        
        # fraction of the mass of the halo that is in form of
        # sub-halos. We have to take this into account while calculating the
        # star formation rate of the central halos. It should be calculated by
        # accounting for this fraction of the subhalo mass in the halo mass
        # central halo mass in this case is (1-f_sub)*mh where mh is the total
        # mass of the halo.
        # for a given halo mass, f_sub is calculated by taking the first moment
        # of the sub-halo mf and and integrating it over all the subhalo masses
        # and dividing it by the total halo mass.
            
        prefact = self.geom_prefact_over_KC # (Nz)
        ncen = self.hod.ncen(theta_hod)
        sfr = self.compute_sfr(theta_sfr)
        sfr_c = ncen * sfr # (Nm, Nz)
        snu = self.compute_snu(theta_snu)   # (Nnu, Nz)
        
        central_emissivity = prefact[None,None,:] * sfr_c[None,:,:] * snu[:,None,:]       
        
        return central_emissivity

    # def emissivity_satellite(self, M, z, theta_sfr, theta_snu):
    #     # analogous calculation for satellites
    #     pass

    def _generate_nu_prime_grid(self):
        """
        Returns nu' = nu*(1+z) grid in units of Hz. 
        
        Useful to pass to any parametric SED model that 
        calculates SED as a function of nu. 
        """
        
        ghz = 1e9 # Giga Hz conversion factor 
        nu_grid =  np.linspace(1e2,4e3,10000)*ghz # sample 10,000 points from 100 to 1000 GHz
        nu_prime_grid = nu_grid[:,np.newaxis] * (1+self.cosmo.z[np.newaxis,:])
        
        return nu_prime_grid
    
    def _load_filters(self):
        filters = []
        for path in self.filter_files:
            data = np.loadtxt(path, usecols=(1,2)) #FIXME: col specifications?
            freq = data[:, 0] * 1e9  # in Hz #FIXME: make this agnostic
            response = data[:, 1]
            filters.append((freq, response))
        return filters

    def _apply_filters(self, sed_vals):
        """
        Applies filter transmission curves to raw SED values to get effective fluxes.
        
        Args:
            sed_vals : (Nz, Nnu) array (SED sampled over nu' for each z)
        
        Returns:
            snu_filtered : (Nfilters, Nz) array
        """
        snu_filtered = []
        
        for freq_filter, response in self.filters:
            flux = SED_to_flux(
                sed_vals.T,
                self.nu_prime.T,  # (Nz, Nnu), assuming self.nu_prime is (Nnu, Nz)
                freq_filter,
                response
            )
            snu_filtered.append(flux)  # shape (Nz,)
        
        return np.array(snu_filtered)  # shape (Nfilters, Nz)
