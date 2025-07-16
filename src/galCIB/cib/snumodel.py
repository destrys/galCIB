#cib/snumodel.py

from .registry import get_snu_model

class SnuModel:
    def __init__(self, 
                 name, 
                 cosmo,
                 nu_prime=None, 
                 m21_fdata="../data/filtered_snu_planck.fits"):
        
        self.name = name
        self.cosmo = cosmo
        self.z = cosmo.z
        if nu_prime is not None: 
            self.nu_prime = nu_prime
        else:
            self.nu_prime = self._generate_nu_prime_grid() # (Nnu, Nz)
        self.model_fn = self._build_model(name, m21_fdata)

    def _build_model(self, name, data_dir):
        factory = get_snu_model(name)
        if name == "Y23":
            return factory(self.nu_prime, self.z)
        elif name == "M21":
            return factory(self.nu_prime, self.cosmo, data_dir)
        else:
            raise ValueError(f"Unknown snu model: {name}")

    def __call__(self, theta_snu):
        return self.model_fn(theta_snu)

    def _generate_nu_prime_grid(self):
        """
        Returns nu' = nu*(1+z) grid in units of Hz. 
        
        Useful to pass to any parametric SED model that 
        calculates SED as a function of nu. 
        """
        
        import numpy as np
        ghz = 1e9
        nu = np.linspace(1e2, 4e3, 10000) * ghz
        return nu[:,None] * (1 + self.z)[None,:] # (Nnu, Nz)
