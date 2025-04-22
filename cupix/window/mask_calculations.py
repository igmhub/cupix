# masking calculations for 1D FFTs
import numpy as np
import numpy.fft as fft


def calculate_masked_power(avg_sqmask_array_fft, theory_power):
    avg_sqmask_array_2fft = np.fft.fft(avg_sqmask_array_fft)
    del avg_sqmask_array_fft
    theory_power_2fft = np.fft.fft(theory_power)
    del theory_power
    masked_power_fft = avg_sqmask_array_2fft * theory_power_2fft
    del avg_sqmask_array_2fft
    del theory_power_2fft
    masked_power = np.fft.ifft(masked_power_fft)
    return masked_power
