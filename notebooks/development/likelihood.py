# %% [markdown]
# # Interfaces between likelihood, data and theory

# %%
from lace.cosmo import camb_cosmo
from cupix.px_data import data_healpix, px_window, px_ztk


# %%
class Emulator(object):
    '''Interface between ForestFlow and the rest of the code'''

    def __init__(self, config):
        '''Setup the emulator given input arguments'''
        print('setting up emulator')
        self.config = config

    def emulate_px_Mpc(self, z, rt, kp, emu_params):
        '''Emulate Px at one z, for several (rt, kp), given emu_params'''
        print('asking ForestFlow to emulate Px')
        # dummy
        px = np.zeros([len(rt), len(kp)])
        return px


# %%
class Theory(object):
    '''Lya theory object to model observed px (with contaminants)'''

    def __init__(self, fid_cosmo, emulator, contaminants):
        '''Setup the theory object given input arguments'''

        print('setting up theory')
        self.fid_cosmo = fid_cosmo
        self.emulator = emulator
        self.contaminants = contaminants

    
    @classmethod
    def from_config(cls, config):
        '''Setup theory from config file (dict)'''

        fid_cosmo = camb_cosmo.get_cosmology_from_dictionary(config)
        emulator = Emulator(config)
        contaminants = None
        return cls(fid_cosmo, emulator, contaminants)

    
    def model_px(self, px_z, like_params):
        '''Given likelihood parameters, and data object, model Px'''

        # px_z containts Px at a given z
        z = px_z.z_bin.mean()
        print(f'modeling px for z = {z}')
        
        # figure out the relevant emulator parameters (if needed)
        emu_params = self.get_emu_params(z, like_params)
        
        # get values of (rt, kp) in Mpc
        rt_Mpc, kp_Mpc = self.get_rt_kp_Mpc(px_z)

        # ask emulator for prediction in Mpc
        px_Mpc = self.emulator.emulate_px_Mpc(z, rt, kp, emu_params)

        # you would here convert to px (Angstroms, arcmin)
        px_arcmin_A = px_Mpc
        
        # then you would add contaminants
        px_arcmin_A *= 1.0
        
        return px_arcmin_A
        

    def get_emu_params(self, z, like_params):
        '''Figure out emulator params given likelihood params'''

        # similar to cup1d.likelihood.get_emulator_calls, but at one z?
        # dummy for now
        emu_params = {'mF': 0.7} 
        return emu_params


    def get_rt_kp_Mpc(self, px_z):
        '''Convert coordinates to Mpc, for a given (not binned) Px_z object'''

        # redshift of the measurement
        z = px_z.z_bin.mean()
        # theta bins (in arcmin), could use t_min, t_max here
        theta_arcmin = [t_bin.mean() for t_bin in px_z.t_bins()]
        # kp bins (in inverse Angstroms)
        kp_A = [k_bin.k for k_bin in px_z.k_bins()]

        # convert to comoving separations with angular comoving distance (dummy)
        rt_Mpc = 1.0 * np.array(theta_arcmin)
        # convert to comoving wavenumbers with Hubble rate (dummy)
        kp_Mpc = 1.0 * np.array(kp_A)
        return rt_Mpc, kp_Mpc


# %%
class Likelihood(object):
    '''Lya theory object to model observed px (with contaminants)'''

    def __init__(self, data, theory):
        '''Setup the likelihood object given input arguments'''
        print('setting up likelihood')
        self.data = data
        self.theory = theory

    
    @classmethod
    def from_config(cls, config):
        '''Setup likelihood from config file (dict)'''

        data = Data(config)
        theory = Theory.from_config(config)
        return cls(data, theory)


    def get_chi2(self, like_params):
        '''Given input parameters, return chi2'''

        # (data-theory) C^{-1} (data-theory)
        return 0.0
    

# %%
class Data(object):
    '''Contains Px measurement, including rebinning if needed'''
    
    def __init__(self, config):
        '''Provide raw (unbinned) data and rebinned data (with cov)'''
    
        if 'fname' in config:
            fname = config['fname']
        else:
            basedir = '/Users/afont/Codes/cupix/data/px_measurements/Lyacolore/'
            fname = basedir + '/px-nhp_41-zbins_4-thetabins_40.hdf5'
            #fname = basedir + '/px-nhp_150-zbins_4-thetabins_40.hdf5'
    
        # collection of Px from multiple healpixels
        archive = data_healpix.HealpixPxArchive(fname)
        print('read Px archive')
    
        # combine healpixels to get raw data (no binning)
        self.raw_px = archive.get_mean_and_cov()
        print('got raw Px data (averaged)')
        
        # rebinned archive, will be used to compute covariance
        rebin_t_factor = config['rebin_t_factor']
        rebin_k_factor = config['rebin_k_factor']
        rebinned_archive = archive.rebin(rebin_t_factor, rebin_k_factor)
        print('got rebinned archive')
    
        # mean of rebinned archive (with cov)
        self.rebinned_px = rebinned_archive.get_mean_and_cov()
        print('got rebinned Px data (averaged)')


# %%
config = {}
emulator = Emulator(config)

# %%
fid_cosmo = camb_cosmo.get_cosmology_from_dictionary(config)
#print(f'H_0 = {fid_cosmo.H0}')
camb_cosmo.print_info(fid_cosmo)
camb_cosmo.dkms_dhMpc(fid_cosmo, z=3)

# %%
theory = Theory(fid_cosmo, emulator, contaminants=None)

# %%
theory_2 = Theory.from_config(config)

# %%
basedir = '/Users/afont/Codes/cupix/data/px_measurements/Lyacolore/'
config = {
    'fname': basedir + '/px-nhp_41-zbins_4-thetabins_40.hdf5',
    'rebin_t_factor': 4, 
    'rebin_k_factor': 4
}
data = Data(config)
print(f'Raw Px has Nt = {len(data.raw_px.t_bins)}, Nk = {len(data.raw_px.k_bins)}')
print(f'Rebinned Px has Nt = {len(data.rebinned_px.t_bins)}, Nk = {len(data.rebinned_px.k_bins)}')

# %%
likelihood = Likelihood(data, theory)

# %%
like_2 = Likelihood.from_config(config)

# %%
like_params = {'mF': 0.8}
likelihood.get_chi2(like_params)

# %%
