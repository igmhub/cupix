
def main():

    cupixpath = get_path_repo('cupix')

    forecast_file = f"{cupixpath}/data/px_measurements/forecast/fcast_best_fit_arinyo_from_p1d_real_bf3_binned_out_px-zbins_4-thetabins_10_w_res_noiseless.hdf5"
    forecast = DESI_DR2(forecast_file, kM_max_cut_AA=0.5, km_max_cut_AA=0.55, theta_min_cut_arcmin=1.0)
    iz = 0
    z = forecast.z[iz]

    true_cosmo_params = {}
    with h5.File(forecast_file) as f:
        for key in f['cosmo_params'].attrs.keys():
            true_cosmo_params[key] = f['cosmo_params'].attrs[key]
    print(true_cosmo_params)

    # translate these to our Lya params
    with h5.File(forecast_file) as f:

        true_lya_params = {}        
        if 'igm_params' in f['P_Z_AM'][f'z_{iz}'].keys():
            igm_params = f['P_Z_AM'][f'z_{iz}']['igm_params'].attrs
        if 'lya_params' in f['P_Z_AM'][f'z_{iz}'].keys():
            lya_params = f['P_Z_AM'][f'z_{iz}']['lya_params'].attrs
            for par in lya_params:
                true_lya_params[par] = lya_params[par]
        
        elif 'ff_emulated_params' in f['P_Z_AM']['z_0'].keys():
            ff_params = f['P_Z_AM']['z_0']['ff_emulated_params'].attrs
            for par in ff_params:
                true_lya_params[par] = ff_params[par]
        else:
            raise ValueError("No IGM or Lya parameters found in the forecast file.")

    print(true_lya_params)

    # use the true cosmology as fiducial
    cosmo = cosmology.Cosmology(cosmo_params_dict=true_cosmo_params)

    # use the true Lya parameters (Arinyo / bias / beta)
    config = true_lya_params | {'verbose': False}
    theory = Theory(z=z, fid_cosmo=cosmo, config=config)
    like = Likelihood(data=forecast, theory=theory, iz=iz)

    # start a bit off
    ini_bias = 1.05 * true_lya_params['bias']
    ini_beta = 0.9 * true_lya_params['beta']

    bias = FreeParameter(
        name='bias',
        min_value=-0.5,
        max_value=-0.01,
        ini_value=ini_bias,
        true_value=true_lya_params['bias'],
        delta=0.01,
        gauss_prior_mean=ini_bias,
        gauss_prior_width=0.05,
        latex_label=r'b_\alpha'
    )
    beta = FreeParameter(
        name='beta',
        min_value=0.1,
        max_value=5.0,
        ini_value=ini_beta,
        delta=0.1,
        true_value=true_lya_params['beta'],
        gauss_prior_mean=ini_beta,
        gauss_prior_width=0.2,   
        latex_label=r'\beta_\alpha'
    )

    #free_params = [bias]
    free_params = [bias, beta]
    for par in free_params:
        print(par.name, par.ini_value, par.true_value)

    post = Posterior(like, free_params, config={'verbose': False})



    Np = len(free_params)
    nwalkers = 4*(Np+2) # 2*(Np+2)
    max_nsteps = 50 + 15 * Np**2 # 50 + 20 * Np**2
    nburnin = 20 + 10 * Np**2
    config={'verbose':True, 'nwalkers':nwalkers, 'max_nsteps': max_nsteps, 'nburnin':nburnin, 'parallel':True}
    print(config)

    post.silence()

    print("Trying to run sampler")

    nthreads = psutil.cpu_count(logical=True)
    ncores = psutil.cpu_count(logical=False)
    assert nthreads == os.cpu_count(), "psutil and os report different number of threads"
    assert nthreads == mp.cpu_count(), "psutil and multiprocessing report different number of threads"

    nthreads_per_core = nthreads // ncores
    nthreads_available = len(os.sched_getaffinity(0))
    ncores_available = nthreads_available // nthreads_per_core
    # let's only use ncores_available to be safe



    samp = Sampler(post, config=config)
    print("Starting pool with %d cores available" % nthreads_available)
    with mp.Pool(processes=nthreads_available) as pool:
        # re-setup sampler with pool
        print("setting up emcee sampler")
        samp.setup_emcee_sampler(config, pool=pool)
        print("emcee sampler set up")

        samp.run_sampler()

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import h5py as h5
    import multiprocessing as mp
    import os
    import psutil

    from lace.cosmo import cosmology
    from cupix.px_data.data_DESI_DR2 import DESI_DR2
    from cupix.likelihood.theory import Theory
    from cupix.likelihood.likelihood import Likelihood
    from cupix.likelihood.free_parameter import FreeParameter
    from cupix.likelihood.posterior import Posterior
    from cupix.likelihood.minimize_posterior import Minimizer
    from cupix.likelihood.sampler import Sampler
    from cupix.utils.utils import get_path_repo

    mp.set_start_method('spawn')
    main()
    print("Sampler finished")