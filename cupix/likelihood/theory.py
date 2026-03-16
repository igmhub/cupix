import copy
import numpy as np

# our modules below
from lace.cosmo import cosmology, rescale_cosmology
from cupix.likelihood.likelihood_parameter import dict_from_likeparam
from cupix.likelihood.model_lya import LyaModel
from cupix.likelihood.model_contaminants import ContaminantsModel



class Theory(object):
    """Likelihood will ask this object for Px predictions"""

    def __init__(self, zs, fid_cosmo=None, config={'verbose':False}):
        """Setup object to compute predictions for the 1D power spectrum.
        Inputs:
            - zs: redshifts that will be evaluated
            - fid_cosmo: fiducial Cosmology object, used for unit conversions, setting the linear power spectrum, etc
            - config: dictionary with other settinsg.
        """

        self.verbose = config.get('verbose', False)
        self.zs = zs

        # this could also be specified from the config 
        if fid_cosmo is None:
            self.fid_cosmo = cosmology.Cosmology()
        else:
            self.fid_cosmo = fid_cosmo

        # setup LyaModel to compute clean P3D, one per z
        self.lya_models = [LyaModel(z, config) for z in self.zs]

        # setup model for systematics / contaminants, one per z
        self.cont_models = [ContaminantsModel(z, config) for z in self.zs]

        # whether to model the different contaminants
        self.include_hcd = config.get('include_hcd', False)
        self.include_metal = config.get('include_metal', False)
        self.include_sky = config.get('include_sky', False)
        self.include_continuum = config.get('include_continuum', False)

        return


    def get_cosmology(self, cosmo=None, params={}):
        """If cosmo is not None, return cosmo. 
           If cosmo is None, use params to compute cosmology"""
        if cosmo is None:
            # figure out whether we can recycle the fiducial cosmology
            if self.fid_cosmo.same_background(cosmo_params=params):
                cosmo = rescale_cosmology.RescaledCosmology(fid_cosmo=self.fid_cosmo,
                                                            new_params_dict=params)
                if self.verbose: print("recycle transfer function")
            else:
                if self.verbose: print("create new CAMB cosmology")
                cosmo = cosmology.Cosmology(cosmo_params_dict=params)
        else:
            if self.verbose: print('cosmology already provided')

        return cosmo


    # currently needed by the likelihood class, keeping the format as is
    def get_px_AA(self, 
        k_AA,
        theta_arcmin,
        zs=None,
        like_params={},
        verbose=None
    ):
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print('inside Theory::get_px_AA')

        # convert list of LikelihoodParameters to dictionary
        params_dict = dict_from_likeparam(like_params)
        if verbose:
            print('params_dict', params_dict)

        return self.get_px_obs(z=zs, theta_arc=theta_arcmin, k_AA=k_AA, 
                               params=params_dict)


    def get_px_obs(self, z, theta_arc, k_AA, cosmo=None, params={}):
        
        # figure out the cosmology to use 
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)

        # hopefully we'll get rid of this soon (we should have a theory per z)
        assert z in self.zs, "Input redshift not found in theory.zs"
        iz = np.argwhere(self.zs == z)[0, 0] 

        if self.verbose:
            print("Evaluating theory at redshift", z)
            print("This will correspond to redshift bin with index", iz)

        if self.verbose:
            print('Theta bins (arcmin)', theta_arc)

        theta_deg = np.atleast_1d(theta_arc) / 60.0
        if self.verbose:
            print('Theta bins (deg)', theta_deg)

        if self.include_hcd:
            # ask for Px (Lya + HCD) 
            px_obs = self.get_px_lya_hcd_obs(iz, theta_arc, k_AA, cosmo, params)
        else:
            # ask for Lya Px 
            px_obs = self.get_px_lya_obs(iz, theta_arc, k_AA, cosmo, params)

        if self.include_metal:
            # compute metals here (silicon auto, and silicon x lya)
            px_metal_auto = self.get_px_metal_auto_obs(iz, theta_arc, k_AA, cosmo, params)
            px_metal_cross = self.get_px_metal_cross_obs(iz, theta_arc, k_AA, cosmo, params)
            px_obs += px_metal_auto + px_metal_cross

        if self.include_sky:
            # compute contamination from sky residuals
            px_sky = self.get_px_sky_obs(iz, theta_arc, k_AA, cosmo, params)
            px_obs += px_sky

        if self.include_continuum:
            # compute multiplicative correction due to continuum fitting
            cont_distortion = self.get_continuum_distortion(iz, k_AA, cosmo, params)
            px_obs *= cont_distortion

        return px_obs
        

    def get_px_lya_obs(self, iz, theta_arc, k_AA, cosmo=None, params={}):

        # figure out the cosmology to use 
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)

        z = self.zs[iz]
        if self.verbose: print('inside Theory::get_px_lya_obs, z=',z)

        # unit conversions to Mpc (where theory lives)
        darc_dMpc = cosmo.get_darc_dMpc(z)
        lr_lya = self.lya_models[iz].lr_lya
        dAA_dMpc = cosmo.get_dAA_dMpc(z, lambda_rest_AA=lr_lya)
        rt_Mpc = theta_arc / darc_dMpc
        kp_Mpc = k_AA * dAA_dMpc

        # compute Px in Mpc
        Px_Mpc = self.get_px_lya_Mpc(iz, rt_Mpc, kp_Mpc, cosmo, params)

        # back to inverse Angstroms
        Px_AA = Px_Mpc * dAA_dMpc

        return Px_AA


    def get_px_lya_hcd_obs(self, iz, theta_arc, k_AA, cosmo=None, params={}):

        # figure out the cosmology to use 
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)

        z = self.zs[iz]
        if self.verbose: print('inside Theory::get_px_lya_hcd_obs, z=',z)

        # unit conversions to Mpc (where theory lives)
        darc_dMpc = cosmo.get_darc_dMpc(z)
        lr_lya = self.lya_models[iz].lr_lya
        dAA_dMpc = cosmo.get_dAA_dMpc(z, lambda_rest_AA=lr_lya)
        rt_Mpc = theta_arc / darc_dMpc
        kp_Mpc = k_AA * dAA_dMpc

        # compute Px in Mpc
        Px_Mpc = self.get_px_lya_hcd_Mpc(iz, rt_Mpc, kp_Mpc, cosmo, params)

        # back to inverse Angstroms
        Px_AA = Px_Mpc * dAA_dMpc

        return Px_AA


    def get_z_metal_auto(self, iz):
        """Given Lya z bin, compute redshift to evaluated Silicon auto"""

        # Lya redshift to use 
        z_lya = self.zs[iz]
        lr_lya = self.lya_models[iz].lr_lya
        lr_metal = self.cont_models[iz].lr_metal
        z_metal = (1+z_lya) * lr_lya / lr_metal - 1
        if self.verbose and False: 
            print('z_lya =',z_lya)
            print('z_metal =',z_metal)

        return z_metal


    def get_z_metal_cross(self, iz):
        """Given Lya z bin, compute redshift to evaluated Silicon x Lya"""

        # Lya redshift to use 
        z_lya = self.zs[iz]
        lr_lya = self.lya_models[iz].lr_lya
        lr_metal = self.cont_models[iz].lr_metal
        z_metal = (1+z_lya) * lr_lya / lr_metal - 1
        z_cross = np.sqrt((1+z_lya)*(1+z_metal)) - 1
        if self.verbose and False: 
            print('z_lya =',z_lya)
            print('z_metal =',z_metal)
            print('z_cross =',z_cross)

        return z_cross


    def get_px_metal_auto_obs(self, iz, theta_arc, k_AA, cosmo=None, params={}):

        # figure out the cosmology to use 
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)

        # redshift to use (not the same as the Lya z)
        z_metal_auto = self.get_z_metal_auto(iz)

        # unit conversions to Mpc (where theory lives)
        darc_dMpc = cosmo.get_darc_dMpc(z_metal_auto)
        lr_metal = self.cont_models[iz].lr_metal
        dAA_dMpc = cosmo.get_dAA_dMpc(z_metal_auto, lambda_rest_AA=lr_metal)
        rt_Mpc = theta_arc / darc_dMpc
        kp_Mpc = k_AA * dAA_dMpc

        # compute Px in Mpc
        Px_Mpc = self.get_px_metal_auto_Mpc(iz, rt_Mpc, kp_Mpc, cosmo, params)

        # back to inverse Angstroms
        Px_AA = Px_Mpc * dAA_dMpc

        return Px_AA


    def get_px_metal_cross_obs(self, iz, theta_arc, k_AA, cosmo=None, params={}):

        # figure out the cosmology to use 
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)

        # redshift to use (not the same as the Lya z)
        z_metal_cross = self.get_z_metal_cross(iz)

        # unit conversions to Mpc (where theory lives)
        darc_dMpc = cosmo.get_darc_dMpc(z_metal_cross)
        lr_metal = self.cont_models[iz].lr_metal
        lr_lya = self.lya_models[iz].lr_lya
        lr_cross = np.sqrt(lr_metal*lr_lya)
        dAA_dMpc = cosmo.get_dAA_dMpc(z_metal_cross, lambda_rest_AA=lr_cross)
        rt_Mpc = theta_arc / darc_dMpc
        kp_Mpc = k_AA * dAA_dMpc

        # compute Px in Mpc
        Px_Mpc = self.get_px_metal_cross_Mpc(iz, rt_Mpc, kp_Mpc, cosmo, params)

        # back to inverse Angstroms
        Px_AA = Px_Mpc * dAA_dMpc

        return Px_AA




    def _compute_px_from_p3d(self,rt_Mpc, kp_Mpc, p3d_func_kmu, kt_Mpc_max):

        # trying to recycle existing Px functionality in ForestFlow
        from forestflow import pcross
        from forestflow.model_p3d_arinyo import coordinates
        @coordinates("k_mu")
        def dummy_p3d_func_kmu(dummy, k, mu, ari_pp=None, new_cosmo_params=None):
            return p3d_func_kmu(k, mu)
        return pcross.Px_Mpc_detailed(
                z=123456789,
                kpar_iMpc=kp_Mpc,
                rperp_Mpc=rt_Mpc,
                p3d_fun_Mpc=dummy_p3d_func_kmu,
                p3d_params={'dummy':123456789},
                max_k_for_p3d=kt_Mpc_max)


    def _compute_lya_hcd_biases(self, kpar, lya_params, hcd_params):
        """Compute scale-dependent bias for Lya + HCD"""

        b_a = lya_params['bias']
        beta_a = lya_params['beta']
        b_H = hcd_params['b_H']
        beta_H = hcd_params['beta_H']
        L_H_Mpc = hcd_params['L_H_Mpc']
        # F_hcd(kpar)
        F_H = np.exp(-kpar*L_H_Mpc)
        # total (scale-dependent) bias
        b_T = b_a + b_H * F_H
        b_beta_T = b_a * beta_a + b_H * beta_H * F_H
        beta_T = b_beta_T / b_T

        return b_T, beta_T


    def _compute_DNL_Arinyo(self, k, mu, linP, lya_params):
        """Compute small-scales correction from Arinyo 2015"""

        # Model the small-scale correction (D_NL in Arinyo-i-Prats 2015)
        delta2 = (1 / (2 * np.pi**2)) * k**3 * linP
        nonlin = delta2 * (lya_params["q1"] + lya_params["q2"] * delta2)
        vel = (k / lya_params["kv_Mpc"]) ** lya_params["av"] * mu ** lya_params["bv"]
        press = (k / lya_params["kp_Mpc"]) ** 2
        D_NL = np.exp(nonlin * (1 - vel) - press)

        return D_NL


    def get_p3d_lya_Mpc(self, iz, k, mu, cosmo=None, params={}):
        z = self.zs[iz]
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        linP = cosmo.get_linP_Mpc(z, k)
        lya_params = self.lya_models[iz].get_lya_params(cosmo, params)
        bias = lya_params['bias']
        beta = lya_params['beta']
        # large-scales power
        p3d = bias**2 * (1 + beta * mu**2)**2 * linP
        # DNL correction from Arinyo 
        DNL = self._compute_DNL_Arinyo(k, mu, linP, lya_params)
        return p3d * DNL


    def get_px_lya_Mpc(self, iz, rt_Mpc, kp_Mpc, cosmo=None, params={}):
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        # function to be passed to compute Px
        def p3d_func(k, mu):
            return self.get_p3d_lya_Mpc(iz, k, mu, cosmo, params)

        # maximum kt_Mpc to use (power should be 0 past that)
        # could ask CAMB object, but pressure is doing this job for you
        # kt_Mpc_max = 5 * lya_params['kp_Mpc']
        kt_Mpc_max = 200.0

        return self._compute_px_from_p3d(rt_Mpc, kp_Mpc, p3d_func, kt_Mpc_max)


    def get_p3d_lya_hcd_Mpc(self, iz, k, mu, cosmo=None, params={}):

        z = self.zs[iz]

        # figure out cosmology to use from input
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        linP = cosmo.get_linP_Mpc(z, k)

        # get the complete list of lya parameters (bias, beta, arinyo)
        # from defaults and input params, potentially using the emulator
        lya_params = self.lya_models[iz].get_lya_params(cosmo, params)
        # get the complete list of hcd parameters (b_H, beta_H, L_H_Mpc)
        hcd_params = self.cont_models[iz].get_hcd_params(params)

        # scale-dependent bias for Lya + HCD
        kpar = k*mu
        b_T, beta_T = self._compute_lya_hcd_biases(kpar, lya_params, hcd_params)
        # large-scales power
        p3d = b_T**2 * (1 + beta_T * mu**2)**2 * linP
        # DNL correction from Arinyo (probably should not apply to HCD...)
        DNL = self._compute_DNL_Arinyo(k, mu, linP, lya_params)
        return p3d * DNL


    def get_px_lya_hcd_Mpc(self, iz, rt_Mpc, kp_Mpc, cosmo=None, params={}):
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        # function to be passed to compute Px
        def p3d_func(k, mu):
            return self.get_p3d_lya_hcd_Mpc(iz, k, mu, cosmo, params)

        # maximum kt_Mpc to use (power should be 0 past that)
        # could ask CAMB object, but pressure is doing this job for you
        # kt_Mpc_max = 5 * lya_params['kp_Mpc']
        kt_Mpc_max = 200.0

        return self._compute_px_from_p3d(rt_Mpc, kp_Mpc, p3d_func, kt_Mpc_max)


    def get_p3d_metal_auto_Mpc(self, iz, k, mu, cosmo=None, params={}):
        # evaluate linP at different z than Lya
        z_metal_auto = self.get_z_metal_auto(iz)

        # figure out cosmology to use from input
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        linP = cosmo.get_linP_Mpc(z_metal_auto, k)

        # get the complete list of metal parameters (b_X, beta_X)
        metal_params = self.cont_models[iz].get_metal_params(params)
        bias = metal_params['b_X']
        beta = metal_params['beta_X']

        # large-scales power
        p3d = bias**2 * (1 + beta * mu**2)**2 * linP

        # suppress power on small scales (help with Hankl)
        kp_Mpc = 10.0
        smooth = np.exp(-(k/kp_Mpc)**2)

        return p3d * smooth


    def get_px_metal_auto_Mpc(self, iz, rt_Mpc, kp_Mpc, cosmo=None, params={}):
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        # function to be passed to compute Px
        def p3d_func(k, mu):
            return self.get_p3d_metal_auto_Mpc(iz, k, mu, cosmo, params)

        # maximum kt_Mpc to use (power should be 0 past that)
        # could ask CAMB object, but pressure is doing this job for you
        # kt_Mpc_max = 5 * lya_params['kp_Mpc']
        kt_Mpc_max = 200.0

        return self._compute_px_from_p3d(rt_Mpc, kp_Mpc, p3d_func, kt_Mpc_max)


    def get_p3d_metal_cross_Mpc(self, iz, k, mu, cosmo=None, params={}):
        # evaluate linP at different z than Lya
        z_cross = self.get_z_metal_cross(iz)

        # figure out cosmology to use from input
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        linP = cosmo.get_linP_Mpc(z_cross, k)

        # get the complete list of lya parameters (bias, beta, arinyo)
        # from defaults and input params, potentially using the emulator
        lya_params = self.lya_models[iz].get_lya_params(cosmo, params)
        b_a = lya_params['bias']
        beta_a = lya_params['beta']

        # get the complete list of metal parameters (b_X, beta_X)
        metal_params = self.cont_models[iz].get_metal_params(params)
        b_X = metal_params['b_X']
        beta_X = metal_params['beta_X']

        # large-scales power
        p3d = b_a * b_X * (1 + beta_a * mu**2) * (1 + beta_X * mu**2) * linP

        # suppress power on small scales (help with Hankl)
        kp_Mpc = 10.0
        smooth = np.exp(-(k/kp_Mpc)**2)

        # scale of silicon oscillations (in log lambda)
        lr_lya = self.lya_models[iz].lr_lya
        lr_metal = self.cont_models[iz].lr_metal
        dX_loglam = np.log(lr_lya / lr_metal)
        # scale in observed Angstroms
        lr_cross = np.sqrt(lr_lya * lr_metal)
        dX_AA = dX_loglam * (1 + z_cross) * lr_cross
        # translate to comoving Mpc
        drp_Mpc = dX_AA / cosmo.get_dAA_dMpc(z_cross, lambda_rest_AA=lr_cross)

        # compute oscillations
        kpar = k * mu
        wiggles = 2 * np.cos(kpar * drp_Mpc)

        return p3d * smooth * wiggles


    def get_px_metal_cross_Mpc(self, iz, rt_Mpc, kp_Mpc, cosmo=None, params={}):
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        # function to be passed to compute Px
        def p3d_func(k, mu):
            return self.get_p3d_metal_cross_Mpc(iz, k, mu, cosmo, params)

        # maximum kt_Mpc to use (power should be 0 past that)
        # could ask CAMB object, but pressure is doing this job for you
        # kt_Mpc_max = 5 * lya_params['kp_Mpc']
        kt_Mpc_max = 200.0

        return self._compute_px_from_p3d(rt_Mpc, kp_Mpc, p3d_func, kt_Mpc_max)


    def get_px_sky_obs(self, iz, theta_arc, k_AA, cosmo=None, params={}):

        # figure out the cosmology to use 
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        z = self.zs[iz]

        # unit conversions to Mpc (where theory lives)
        darc_dMpc = cosmo.get_darc_dMpc(z)
        lr_lya = self.lya_models[iz].lr_lya
        dAA_dMpc = cosmo.get_dAA_dMpc(z, lambda_rest_AA=lr_lya)
        rt_Mpc = theta_arc / darc_dMpc
        kp_Mpc = k_AA * dAA_dMpc
        
        # compute Px in Mpc
        Px_Mpc = self.get_px_sky_Mpc(iz, rt_Mpc, kp_Mpc, cosmo, params)

        # back to inverse Angstroms
        Px_AA = Px_Mpc * dAA_dMpc

        return Px_AA 


    def get_px_sky_Mpc(self,iz, rt_Mpc, kp_Mpc, cosmo=None, params={}):

        # figure out the cosmology to use
        cosmo = self.get_cosmology(cosmo=cosmo, params=params)
        z = self.zs[iz]

        # unit conversions (should be fast) 
        darc_dMpc = cosmo.get_darc_dMpc(z)
        lr_lya = self.lya_models[iz].lr_lya
        dAA_dMpc = cosmo.get_dAA_dMpc(z, lambda_rest_AA=lr_lya)

        # 1 at theta=0, 0 at large angular separations
        theta_arc = rt_Mpc * darc_dMpc
        xi_noise = self.cont_models[iz].get_xi_noise(theta_arc)

        # get b_noise parameter (using default and input params)
        sky_params = self.cont_models[iz].get_sky_params(params)
        b_noise_Mpc = sky_params['b_noise_Mpc']

        # pixel width in Angstroms
        x_AA = 0.8
        x_Mpc = x_AA / dAA_dMpc

        # pixel smoothing 
        x = x_Mpc * kp_Mpc / 2.0
        smooth = (np.sin(x)/x)**2

        Px_sky = b_noise_Mpc * np.outer(xi_noise, smooth)

        return Px_sky

