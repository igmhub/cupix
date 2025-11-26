# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: pcross
#     language: python
#     name: python3
# ---

# %%
import sys
import numpy as np
from cupix.likelihood.lya_theory import set_theory
from cupix.likelihood.forestflow_emu import FF_emulator
from cupix.likelihood.input_pipeline import Args
from lace.cosmo import camb_cosmo
import matplotlib.pyplot as plt
from cupix.likelihood.pipeline import set_like, Pipeline
from cupix.likelihood.fitter import Fitter
from cupix.likelihood.pipeline import set_Px


# %%
# %load_ext autoreload
# %autoreload 2

# %%
args = Args(data_label='lyacolore')
args.set_baseline()

# %% [markdown]
# Load the emulator

# %%
# Load emulator
z = np.array([2.2])
omnuh2 = 0.0006
mnu = omnuh2 * 93.14
H0 = 67.36
omch2 = 0.12
ombh2 = 0.02237
As = 2.1e-9
ns = 0.9649
nrun = 0.0
w = -1.0
omk = 0
cosmo = {
    'H0': H0,
    'omch2': omch2,
    'ombh2': ombh2,
    'mnu': mnu,
    'omk': omk,
    'As': As,
    'ns': ns,
    'nrun': nrun,
    'w': w
}
sim_cosmo = camb_cosmo.get_cosmology_from_dictionary(cosmo)
cc = camb_cosmo.get_camb_results(sim_cosmo, zs=z, camb_kmax_Mpc=1000)
ffemu = FF_emulator(z, cosmo, cc)

# %% [markdown]
# Set the arguments

# %%
args.emu_cov_factor = None
args.use_pk_smooth = False
args.rebin_k = 1
args.n_steps = 3
args.emulator = ffemu
args.n_tau = 1
args.n_sigT = 0
args.n_gamma = 0
args.n_kF = 0
args.n_d_dla = 0
args.n_s_dla = 0
args.n_sn = 0
args.n_agn = 0
args.n_res = 0
args.nwalkers = 25

# %% [markdown]
# Setup the pipeline

# %%
pipe = Pipeline(args)

# %%
p0 = np.array(list(pipe.fitter.like.fid["fit_cube"].values()))
pipe.run_minimizer(p0)

# %%
from cupix.likelihood.plotter import Plotter # need to change to cupix later

pipe.plotter = Plotter(
    pipe.fitter, save_directory=pipe.fitter.save_directory
    )


# %%
Nfft = 30
perf_res = np.ones(Nfft)
w = np.ones(Nfft)
W = np.fft.fft(w)*np.conj(np.fft.fft(w))
print(W)
oldnorm = np.fft.ifft(np.fft.fft(perf_res**2)*np.fft.fft(W))
newnorm = perf_res**2*W
print(oldnorm, newnorm)

# %%
W

# %%
pipe.plotter.plots_minimizer()

# %%
pipe.mle_values = pipe.fitter.get_best_fit(stat_best_fit="mle")
     63 self.like_params = self.fitter.like.parameters_from_sampling_point(
     64     self.mle_values


# %%
pipe.fitter.like.plot_px(
    values =pipe.mle_values,
    plot_every_iz=1,
    return_all=True,
    show=False,
    zmask=None
)

# %% [markdown]
# Set the likelihood

# %%
Px_data = set_Px(args)
# remove the zero
print(Px_data.Pk_AA.shape, Px_data.k_AA.shape, Px_data.window.shape)

Px_data.k_AA = Px_data.k_AA[:,1:200]
Px_data.Pk_AA = Px_data.Pk_AA[:,:, 1:200]
Px_data.window = Px_data.window[:,:, 1:200, 1:200]
print(Px_data.Pk_AA.shape, Px_data.k_AA.shape, Px_data.window.shape)

# %%
p0 = np.array(list(likelihood.fid["fit_cube"].values()))
pipe.run_minimizer(p0)

# %% [markdown]
# Set the theory

# %%
args = Args(data_label='lyacolore')
args.set_baseline()
theory_AA = set_theory(args, ffemu, k_unit='iAA')
theory_AA.set_fid_cosmo(z)

# %%
args.fix_cosmo

# %%
dir(args)

# %% [markdown]
# ## Set the data

# %%
from cupix.likelihood.pipeline import set_Px
Px_data = set_Px(args)
# remove the zero
print(Px_data.Pk_AA.shape, Px_data.k_AA.shape, Px_data.window.shape)

Px_data.k_AA = Px_data.k_AA[:,1:200]
Px_data.Pk_AA = Px_data.Pk_AA[:,:, 1:200]
Px_data.window = Px_data.window[:,:, 1:200, 1:200]
print(Px_data.Pk_AA.shape, Px_data.k_AA.shape, Px_data.window.shape)

# %% [markdown]
# Note: it seems to matter whehter we keep all the k modes or only the positive ones, for the window matrix calculations

# %%
for i in range(len(Px_data.thetabin_deg[0])):
    print(f"theta bin {Px_data.thetabin_deg[0][i]}")
    plt.plot(Px_data.k_AA[0], Px_data.Pk_AA[0,i], label='data')
plt.xlim([0,0.8])

# %%
out_AA = theory_AA.get_px_AA(
        zs = z,
        k_AA=Px_data.k_AA,
        theta_bin_deg=Px_data.thetabin_deg,
        window_function=Px_data.window,
        return_blob=False
    )

# %%
for iz, zbin in enumerate(out_AA):
    print("z=", z[iz])
    if iz==0:
        linestyle='dashed'
    else:
        linestyle='solid'
    
    for itheta, theta in enumerate(zbin):
        # print(out[iz][itheta])
        plt.plot(Px_data.k_AA[iz], out_AA[iz][itheta], label=f'theta=[{(Px_data.thetabin_deg[iz][itheta]*60)[0]:.0f}, {(Px_data.thetabin_deg[iz][itheta]*60)[1]:.0f}] arcmin, z={z[iz]:.1f}', linestyle=linestyle)
        # plot the data
        plt.plot(Px_data.k_AA[iz], Px_data.Pk_AA[iz, itheta], markersize=3, color='black', alpha=0.5)
plt.legend()
plt.xlim([0,0.8])
plt.ylim([-0.01,.16])
plt.ylabel('$P(k)~[\AA]$')
plt.xlabel(r'$k~[\AA^{-1}]$')

# %%
Px_data.cov_Pk_AA.shape

# %%
args.emu_cov_factor = None
args.use_pk_smooth = False
args.rebin_k = 1
Px_data.Pksmooth_kms = None
Px_data.full_Pk_AA = None

# %%
likelihood = set_like(Px_data, ffemu, args, data_hires=None)

# %%
args.n_steps = 100

# %%
fitter =  Fitter(
            like=likelihood,
            rootdir=None,
            nburnin=args.n_burn_in,
            nsteps=args.n_steps,
            parallel=args.parallel,
            explore=args.explore,
            fix_cosmology=args.fix_cosmo,
        )


# %%
args.emulator = ffemu

# %%
pipe = Pipeline(args)

# %%
pipe.fitter.like.data.k_AA.shape

# %%
p0 = np.array(list(likelihood.fid["fit_cube"].values()))
pipe.run_minimizer(p0)

# %%
Pipeline.run_minimizer()

# %%
