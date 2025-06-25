#from galCIB.galaxy import get_hod_model
from scipy.integrate import simpson
import numpy as np 

class AnalysisModel:
    def __init__(self, cosmology, survey, pk3d, hmalpha=1):
        """
        Initializes the model with fixed cosmology and survey properties.

        Args:
            cosmology: Cosmology object (with Mh, z, k, P_lin, etc.)
            survey: Survey object (with z, pz, compute_Wg, etc.)
            hod_name: Name of the default HOD model to use
            profile_type: e.g., "nfw" or "mixed"
        """
        
        self.cosmo = cosmology
        self.survey = survey
        self.hod_name = pk3d
        
        self.geom_factor = self.cosmo.geom_factor_c_ell
        self.Wg = survey.Wg
        
        self.pkgg_2h = pk3d.Pgg_2h
        self.pkgg_1h = pk3d.Pgg_1h
        
        pkz_gg_interp_2h = self._kPk_interpolator(self.pkgg_2h)
        pkz_gg_interp_1h = self._kPk_interpolator(self.pkgg_1h)
        
        self.cl_gg_2h = self.compute_cl(pkz_gg_interp_2h,self.Wg,self.Wg)
        self.cl_gg_1h = self.compute_cl(pkz_gg_interp_1h,self.Wg,self.Wg)
    
    def _kPk_interpolator(self, pk):
        """
        k and Pk need to be interpolated on the same grid 
        to calculate C_ell. The grid is:
        k = (ell+0.5)/chi(z)
        
        Returns:
            kz_grid : (Nell, Nz)
            Pk_grid : (Nell, Nz)
        """
        
        self.kz_grid = (self.survey.ells[:,np.newaxis]+0.5)/self.cosmo.chi[np.newaxis,:]
        
        # interpolate pk 
        
        pk_interp = np.zeros((len(self.survey.ells),
                              len(self.survey.z)))
        for zidx in range(len(self.survey.z)):
            pk_interp[:,zidx] = np.interp(self.kz_grid[:,zidx],
                                          self.cosmo.k,
                                          pk[:,zidx])
            
        return pk_interp
        
    def compute_cl(self,pkz_xy, Wx, Wy):
        """
        Return C_ell gg using Limber. 
        
        Args:
            pkz_xy : interpolated P(k) of X and Y fields.
            Wx : Window of X
            Wy : Window of Y
        """
        
        integrand = self.geom_factor*Wx*Wy*pkz_xy
        cl = simpson(integrand,x=self.survey.z,axis=-1)
        
        return cl

    # def loglike(self, theta, data, cov):
    #     """
    #     Compute log-likelihood for given theta and data.
    #     """
    #     cl_model = self.compute_cl_gg(theta)
    #     chi2 = np.dot((cl_model - data), np.linalg.solve(cov, cl_model - data))
    #     return -0.5 * chi2
