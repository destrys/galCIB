"""
This sets up the galaxy HOD class that calculates relevant properties such as 
HOD models. It has a few default HOD models and can also take user-defined models. 
"""

class HODModel:
    """
    Container class for a Halo Occupation Distribution (HOD) model.
    
    Attributes
    ----------
    name : str
        Unique identifier for the model.
    ncen_fn : callable
        Function that returns ⟨N_cen(M, z)⟩ given M, z, and theta.
    nsat_fn : callable
        Function that returns ⟨N_sat(M, z)⟩ given M, z, and theta.
    """

    def __init__(self, name, ncen_func, nsat_func, z):
        self.name = name
        self._ncen_func = ncen_func
        self._nsat_func = nsat_func
        self.z = z
        self.z_over_1plusz = z/(1+z)

    def ncen(self, log10Mh, theta_cen):
        return self._ncen_func(log10Mh, theta_cen, 
                               self.z, self.z_over_1plusz)

    def nsat(self, log10Mh, theta_sat, **kwargs):
        return self._nsat_func(log10Mh, theta_sat, 
                               self.z, self.z_over_1plusz,
                               **kwargs)

    def evaluate(self, theta_cen, theta_sat, log10Mh, z=None):
        """Evaluate both N_cen and N_sat for given halo mass 
        and redshift grids."""
        ncen_vals = self.ncen(log10Mh, theta_cen, z)
        nsat_vals = self.nsat(log10Mh, theta_sat, z, ncen=ncen_vals)
        return {
            "N_cen": ncen_vals,
            "N_sat": nsat_vals
        }