import sys
from forestflow import pcross
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class LyaP3D():
    def __init__(
            self,
            z,
            P3D_model,
            P3D_coeffs,
            kp_Mpc=0.7,
            Si_contam=False,
            contam_coeffs={},
            Arinyo=None,

    ):
        
        self.z = z
        self.Si_contam = Si_contam
        self.kp_Mpc = kp_Mpc
        self.P3D_coeffs = P3D_coeffs
        self.P3D_model = P3D_model
        if Si_contam:
            if not contam_coeffs:
                raise ValueError("Si contamination coefficients must be provided if Si contamination is enabled.")
            self.contam_coeffs = contam_coeffs
        self.arinyo = Arinyo

    def model_Px(self, kpar_Mpc, rperp_Mpc):
        # Code won't work if kpar has a zero
        if 0 in  kpar_Mpc:
            sys.exit('kpar array must not have a zero')

        # make sure that k_Mpc, rperp_Mpc have the same length
        Nz = len(self.z)
        assert kpar_Mpc.shape[0] == Nz, "kpar_Mpc must have length (Nz)"
        assert rperp_Mpc.shape[0] == Nz, "rperp_Mpc must have length (Nz)"

        # ensure that all coefficients are the same shape arrays
        for key in self.P3D_coeffs.keys():
            if self.P3D_coeffs[key].shape != (Nz,):
                raise ValueError(f"Arinyo coefficient {key} must be a 1D array of shape ({Nz},)")
        
        if self.Si_contam:
            Px_pred_Mpc = Px_Mpc_withSiIII(self.z, kpar_Mpc, rperp_Mpc, self.P3D_model, P3D_params=self.P3D_coeffs, Si_coeffs=self.contam_coeffs, Arinyo=self.arinyo)
        else:
            print("Sending in the arinyo coeffs", self.P3D_coeffs)
            # Px_pred_Mpc = self.arinyo.Px_Mpc(self.z, k_Mpc, arinyo_coeffs, **{'rperp_choice':theta_Mpc})
            print(kpar_Mpc.shape)
            Px_pred_Mpc = pcross.Px_Mpc(self.z, kpar_Mpc, rperp_Mpc, self.P3D_model, P3D_params=self.P3D_coeffs)
            
        if np.any(np.isnan(Px_pred_Mpc)):
            print("NaN encountered in Px prediction!")
        return Px_pred_Mpc
    

# copied from ForestFlow
def P1D_Mpc(P3D_Mpc, z, ln_k_perp, kpars, P3D_mode="pol", P3D_params={}):
    """
    Compute P1D by integrating P3D in terms of ln(k_perp) using a fast method.
    Replicates the function in model_p3d_arinyo.py, but with flexibility for the style of P3D input.
    Parameters:
        P3D (function): Function that takes arguments
        z (float): Redshift.
        ln_k_perp (array-like): Array of natural logarithms of the perpendicular wavenumber.
        kpars (array-like): Array of parallel wavenumbers.
        parameters (dict, optional): Additional parameters for the model. Defaults to {}.
    Optional parameters:
        P3D_mode: 'pol' or 'cart' for polar or cartesian. 'pol' assumes that the function takes parameters z and an array of k and mu. 'cart' assumes that the parameters are z, kpar and kperp, both arrays.

    Returns:
        array-like: Computed values of P1D.
    """

    # get interval for integration
    dlnk = ln_k_perp[1] - ln_k_perp[0]

    # get function to be integrated
    # it is equivalent of the inner loop of _P1D_lnkperp
    k_perp = np.exp(ln_k_perp)
    k = np.sqrt(kpars[np.newaxis, :] ** 2 + k_perp[:, np.newaxis] ** 2)
    mu = kpars[np.newaxis, :] / k
    k = k.swapaxes(0, 1)
    mu = mu.swapaxes(0, 1)
    fact = (1 / (2 * np.pi)) * k_perp[:, np.newaxis] ** 2
    fact = fact.swapaxes(0, 1)
    if P3D_mode == "pol":
        p3d_fix_k_par = P3D_Mpc(z, k, mu, P3D_params) * fact
    elif P3D_mode == "cart":
        # tile
        kperp2d = np.tile(k_perp[:, np.newaxis], len(kpars)).T  # mu grid for P3D
        kpar2d = np.tile(kpars[:, np.newaxis], len(k_perp))
        p3d_fix_k_par = P3D_Mpc(z, kpar2d, kperp2d, P3D_params) * fact

    # perform numerical integration
    p1d = simpson(p3d_fix_k_par, ln_k_perp, dx=dlnk, axis=1)

    return p1d




def kaiser(bias1, bias2, beta_rsd1, beta_rsd2, mu):
    """Returns the Kaiser factor for multiplying linear power for a given bias, beta and mu.
    Parameters:
        bias (float): Bias factor.
        beta_rsd (float): Redshift space distortion parameter.
        mu (float): 2D array, Cosine of the angle between the line of sight and the wavevector.
    Returns:
        float: Kaiser factor.
    """
    return (
        bias1
        * bias2
        * (1 + beta_rsd1 * mu**2 + beta_rsd2 * mu**2 + beta_rsd1 * beta_rsd2 * mu**4)
    )


def Px_Mpc_withSiIII(
    z,
    kpar_iMpc,
    rperp_Mpc,
    P3D_Mpc,
    P3D_mode="pol",
    Si_coeffs={},
    P3D_params={}, 
    Arinyo=None
):
    """
    Compute the cross-power spectrum P_cross(k_parallel, r_perp) using a Hankel transform
    for pairs of lines-of-sight separated by transverse distance r_perp, given a 3D power
    spectrum model P3D_Mpc.

    This is the low-level implementation called by `Px_Mpc`. It allows explicit control
    over the k_perp grid, interpolation, and the transition to the 1D power spectrum at
    very small r_perp.

    Parameters
    ----------
    z : float or array-like of shape (Nz,)
        Redshift(s) at which to evaluate the cross-power spectrum.
    kpar_iMpc : array-like
        Parallel wavenumbers k_parallel in units of Mpc⁻¹.
        - Shape can be (Nk,) for a single redshift or (Nz, Nk) for multiple redshifts.
    rperp_Mpc : array-like
        Perpendicular separations r_perp (in Mpc) at which to evaluate the cross-power spectrum.
        - Shape can be (Nr,) for a single redshift or (Nz, Nr) for multiple redshifts.
    P3D_Mpc : callable
        Function returning the 3D power spectrum in Mpc units.
        - If `P3D_mode='pol'`, called as: `P3D_Mpc(z, k, mu, params)`
        - If `P3D_mode='cart'`, called as: `P3D_Mpc(z, kpar, kperp, params)`
    P3D_mode : {'pol', 'cart'}, optional
        Determines how P3D_Mpc is evaluated. Default is 'pol'.
    min_kperp, max_kperp : float, optional
        Minimum and maximum k_perp (Mpc⁻¹) used for the Hankel transform. Default: 1e-7, 1e3.
    nkperp : int, optional
        Number of k_perp points for the Hankel transform. Controls the output r_perp sampling.
        Default is 2**11 (~2048).
    interpmin, interpmax : float, optional
        r_perp range (in Mpc) over which to smoothly interpolate between the 3D cross-power
        and the 1D power spectrum to avoid divergences. Default: 0.005–0.2 Mpc.
    fast_transition : bool, optional
        If True, directly replaces small-r_perp values with P1D without interpolation.
        This is faster but introduces a discontinuity. Default is False.
    P3D_params : dict or list of dicts, optional
        Extra keyword parameters for P3D_Mpc.
        - Dict is broadcast to all z values (with a warning if multi-z).
        - List of dicts must have length Nz.

    Returns
    -------
    Px_pertheta_perz : ndarray
        Cross-power spectrum P_cross in Mpc units evaluated at each input r_perp and k_parallel.
        - Shape is (Nz, Nr, Nk) for multi-z input, or (Nr, Nk) for single z.
    """
    import hankl

    beta_SiIII=None
    k_p_SiIII=None
    bias_SiIII=-9.79e-3 # keep this default for now
    for key in Si_coeffs.keys():
        if key=='bias_SiIII':
            bias_SiIII = Si_coeffs[key]
        elif key=='beta_SiIII':
            beta_SiIII = Si_coeffs[key]
        elif key=='k_p_SiIII':
            k_p_SiIII = Si_coeffs[key]

    bias_alpha = P3D_params["bias"]
    beta_alpha = P3D_params["beta"]
    if beta_SiIII is None:
        beta_SiIII = beta_alpha
    if k_p_SiIII is None:
        k_p_SiIII = P3D_params["kp"]
    # make everything numpy arrays
    kpar_iMpc = np.atleast_1d(kpar_iMpc)
    z_input_type = type(z)
    z = np.atleast_1d(z)
    rperp_Mpc = np.atleast_1d(rperp_Mpc)
    if 0 in kpar_iMpc:
        raise ValueError("kpar list must not contain zero.")
    Nz = len(z)
    if Nz > 1 and kpar_iMpc.ndim == 1:
        kpar_iMpc = np.tile(
            kpar_iMpc, (Nz, 1)
        )  # assume kpar_iMpc is the same for all z
    if Nz > 1 and rperp_Mpc.ndim == 1:
        rperp_Mpc = np.tile(
            rperp_Mpc, (Nz, 1)
        )  # assume rperp_Mpc is the same for all z
    
    if Nz == 1 and kpar_iMpc.ndim == 1:
        # convert to 2d (first dimension is z, second is kpar)
        kpar_iMpc = np.array([kpar_iMpc])
        rperp_Mpc = np.array([rperp_Mpc])
    # ensure that all arrays now have the same first axis
    assert (len(z) == kpar_iMpc.shape[0]
    ), f"Number of redshifts ({len(z)}) does not match number of kpar values ({kpar_iMpc.shape[0]})."
    assert (len(z) == rperp_Mpc.shape[0]
    ), f"Number of redshifts ({len(z)}) does not match number of rperp values ({rperp_Mpc.shape[0]})."
    
    nkpar = kpar_iMpc.shape[1]
    
    print("Here")
    
    # understand what is passed to P3D_params. Turn P3D_params into a list of dictionaries if it is not already
    if P3D_params:
        if isinstance(P3D_params, dict):
            if Nz == 1:
                P3D_params_byz = [P3D_params]
            elif Nz > 1:
                P3D_params_byz = []
                for iz in range(Nz):
                    P3Dsubdictz = {}
                    for key in P3D_params.keys():
                        if isinstance(P3D_params[key], list) or isinstance(
                            P3D_params[key], np.ndarray
                        ):
                            assert (
                                len(P3D_params[key]) == Nz
                            ), f"Parameter {key} must be a list of length {Nz} if z is an array."
                            P3Dsubdictz[key] = P3D_params[key][iz]
                        else:
                            P3Dsubdictz[key] = P3D_params[key]
                            if iz == 0:
                                print(
                                    f"Warning: Number of input z values ({Nz}) does not match the number of values input for parameter {key}. Calculating model with {key} = {P3D_params[key]} for all z."
                                )
                            # assume the parameter is the same for all z, with a warning
                    P3D_params_byz.append(P3Dsubdictz)
        elif isinstance(P3D_params, list):
            # make sure each element is a dictionary
            for P3D_par in P3D_params:
                if not isinstance(P3D_par, dict):
                    raise ValueError("P3D_params must be a list of dictionaries.")
            # make sure the length of the list matches the number of z values
            assert (
                len(P3D_params) == Nz
            ), f"Number of z values ({Nz}) does not match the number of P3D_params dictionaries ({len(P3D_params)})."
            P3D_params_byz = P3D_params
    else:
        raise Warning(
            "P3D_params is empty. Assuming no parameters are needed for P3D_Mpc."
        )
    kperps = np.logspace(np.log10(10.0**-7), np.log10(10.0**3), 2**11)
    Px_pertheta_perz = []
    
    
    for iz in range(Nz):
        # tile

        kperp2d = np.tile(kperps[:, np.newaxis], nkpar)  # mu grid for P3D
        kpar2d = np.tile(kpar_iMpc[iz][:, np.newaxis], 2**11).T
        k2d = np.sqrt(kperp2d**2 + kpar2d**2)
        mu2d = kpar2d / k2d

        if P3D_mode == "cart":
            # assume P3D_Mpc is a function of (z, kpar, kperp)
            P3D_eval_alpha = P3D_Mpc(z[iz], kpar2d, kperp2d, P3D_params_byz[iz])
        elif P3D_mode == "pol":
            # assume P3D_Mpc is a function of (z, k, mu)
            P3D_eval_alpha = P3D_Mpc(z[iz], k2d, mu2d, P3D_params_byz[iz])

        P1D = P1D_Mpc(
            P3D_Mpc,
            z[iz],
            np.linspace(np.log(0.001), np.log(100), 99),
            kpar_iMpc[iz],
            P3D_mode,
            P3D_params_byz[iz],
        )  # get P1D. Later I need to make this include Silicon too.

        # get the P3D for SiIII
        kaiser_SiIII = kaiser(
            bias_SiIII[iz], bias_SiIII[iz], beta_SiIII[iz], beta_SiIII[iz], mu2d
        ) * np.exp(-((k2d / k_p_SiIII[iz]) ** 2))  # include a pressure cutoff
        # get the linear power at this redshift
        P3D_lin = Arinyo.linP_Mpc(z[iz], k2d)
        P3D_eval_SiIII = kaiser_SiIII * P3D_lin
        # get the cross term
        kaiser_cross = kaiser(bias_alpha[iz], bias_SiIII[iz], beta_alpha[iz], beta_SiIII[iz], mu2d)
        P3D_cross = kaiser_cross * P3D_lin

        # combine
        Delta_r_Si_Lya = 20  # Mpc
        P3D_eval = (
            P3D_eval_alpha + P3D_eval_SiIII + P3D_cross * np.cos(kpar2d * Delta_r_Si_Lya)
        )

        Px_per_kpar = []

        for ik, kpar in enumerate(
            kpar_iMpc[iz]
        ):  # for each value of k parallel to evaluate Px at
            P3D_kpar = P3D_eval[:, ik]  # get the P3D
            func = P3D_kpar * kperps
            rperp, LHS = hankl.FFTLog(
                kperps, func, q=0, mu=0
            )  # returns an array of log-spaced rperps, and the Hankel Transform
            Px = LHS / rperp / (2 * np.pi)  # Divide out by remaining factor to get Px
            # transition

            # replace the values left of the minimum
            replace = rperp < (0.02)
            # return the P1D result for that kpar
            Px[replace] = P1D[ik]
            # between rperp = 0.02 and 0.08, interpolate from P1D to Px values
            idxmin = (np.abs(rperp - 0.02)).argmin()
            idxmax = (np.abs(rperp - 0.08)).argmin()
            rperps_interp = rperp[idxmin:idxmax]
            Px_tointerp = np.delete(Px, np.arange(idxmin, idxmax))
            rperp_tointerp = np.delete(rperp, np.arange(idxmin, idxmax))
            interpmin_id = np.abs(rperp_tointerp - .005).argmin()
            interpmax_id = np.abs(rperp_tointerp - 0.2).argmin()
            cs = CubicSpline(
                rperp_tointerp[interpmin_id:interpmax_id],
                Px_tointerp[interpmin_id:interpmax_id],
            )
            Px_interpd = cs(rperps_interp)
            Px = np.insert(Px_tointerp, idxmin, Px_interpd)
            # get an interpolator function that returns Px at the user-requested values for rperp
            Px_func = CubicSpline(rperp, Px)
            Px_per_kpar.append(Px_func(rperp_Mpc[iz]))
        Px_pertheta_perz.append(np.asarray(Px_per_kpar).T)
    Px_pertheta_perz = np.asarray(Px_pertheta_perz)
    # return the cross-power spectrum in the same shape as z was input.
    # if 1 z was input as a float, return a 2D array (Nr, Nk)
    # if 1 or more z was input as an array, return a 3D array (Nz, Nr, Nk)
    
    if Nz == 1:
        # check the input type
        if z_input_type == float:
            # return a 2D array (Nr, Nk)
            Px_pertheta_perz = Px_pertheta_perz.squeeze()
    return Px_pertheta_perz

