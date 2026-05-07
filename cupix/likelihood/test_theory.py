import numpy as np


class TestTheory(object):
    """Fake theory to debug Px measurements"""

    def __init__(self, z, config={}):
        self.z = z
        self.input_power = get_input_power(config)


    def get_px_obs(self, theta_arc, k_AA, cosmo=None, params={}):

        Nt=len(theta_arc)
        Nk=len(k_AA)
        print('in get_px_obs, sizes', Nt, Nk)

        true_px = self.input_power.get_true_px(k_AA)
        print('test true shape', true_px.shape)
        px_obs = np.tile(true_px, (Nt, 1))
        print('test px_obs shape', px_obs.shape)

        return px_obs


def get_input_power(config):
    P0 = config.get('P0', 0.5)
    k0 = config.get('k0', 0.1)
    kF = config.get('kF', 1.0)
    f_px = config.get('f_px', 0.7)
    return InputPower(P0=P0, k0=k0, kF=kF, f_px=f_px)


# the class below has been copied from 
#       https://github.com/sindhusatyavolu/Lya_Px/blob/main/src/Lya_Px/input_power.py
# that was itself copied from
#       https://github.com/andreufont/fourier_playground/blob/master/fft_masking/input_power.py

class InputPower(object):
    """Setup an initial power object given input parameters"""


    def __init__(self,P0=1.0,k0=1,kF=10,f_px=0.2):
        """Define here your favorite power spectrum."""

        self.P0=P0
        self.k0=k0
        self.kF=kF
        self.f_px=f_px


    def get_true_p1d(self,k):
        """Evaluate P1D at input wavenumbers k"""

        return self.get_power(k,is_px=False)
        

    def get_true_px(self,k):
        """Evaluate PX at input wavenumbers k"""

        return self.get_power(k,is_px=True)


    def get_power(self,k,is_px=False):
        """Evaluate power spectrum at input wavenumbers k"""

        if np.any(k<0):
            raise ValueError('InpurPower shold receive non-negative wavenumbers')

        # white noise at low-k
        P = self.P0*np.ones_like(k)

        # small enhancement at low-k
        P *= (1+k/self.k0)
        
        # suppressed at k0
        P *= 1/(1+(k/self.k0)**2)

        # further suppressed with a Gaussian at kF=10
        P *= np.exp(-(k/self.kF)**2)

        # PX will be a scaled version of P1D, for now
        if is_px:
            P *= self.f_px
        
        return P


