# masking calculations for 1D FFTs
import numpy as np

# def calculate_W(weights):
#     '''
#     calculates the average FFT of the weights.
#     weights (np.ndarray): Weights in real-pixel space.
#         N_q x N array, where N_q is the number of quasars and N is the length of the pixel grid
#     Returns:
#     W (np.ndarray): Average of the Fourier-transformed weights-magnitude-squared, vector of length N
#     '''
#     w = np.fft.fft(weights, axis=1)
#     return np.sum((w.real**2 + w.imag**2), axis=0)/w.shape[0]
    
    
# def window_matrix_from_pixel_weights(weights, resolution, L):
#     '''
#     weights (np.ndarray): array N_q [number of quasar spectra] x N [Number of FFT pixels],
#          the real-space pixel weights for each skewer
#     resolution (np.ndarray): Fourier-space resolution function evaluated at pixel coordinates; vector length N
#     L (float): length of skewers

#     Returns:
#     window_matrix (np.ndarray): NxN window matrix
#     estnorm (np.ndarray): vector length N, normalization for the estimated P1D (after averaging over N_q quasars)
#     W (np.ndarray): vector length N, average FFT of the weights 
#     '''
#     W = calculate_W(weights)
#     R2 = resolution.real**2 + resolution.imag**2
#     denom = np.absolute(np.fft.ifft(np.fft.fft(W)* np.fft.fft(R2)))
#     estnorm = np.absolute(L/denom)
#     N = estnorm.size
#     window_matrix = np.zeros((N,N))
#     for m in range(N):
#         for n in range(N):
#             window_matrix[m,n] = W[m-n]*R2[n] / denom[m]
#     return window_matrix, estnorm, W

# def window_matrix_from_fft_weights(W, resolution, L):
#     '''
#     W (np.ndarray): vector length N, average FFT of the weights 
#     resolution (np.ndarray): Fourier-space resolution function evaluated at pixel coordinates; vector length N
#     L (float): length of skewers

#     Returns:
#     window_matrix (np.ndarray): NxN window matrix
#     estnorm (np.ndarray): vector length N, normalization for the estimated P1D (after averaging over N_q quasars)
#     '''
#     R2 = resolution.real**2 + resolution.imag**2
#     denom = np.absolute(np.fft.ifft(np.fft.fft(W)* np.fft.fft(R2)))
#     estnorm = np.absolute(L/denom)
#     N = estnorm.size
#     window_matrix = np.zeros((N,N))
#     for m in range(N):
#         for n in range(N):
#             window_matrix[m,n] = W[m-n]*R2[n] / denom[m]
#     return window_matrix, estnorm

def convolve_window(window_matrix, model):
    '''
    Calculate the prediction for weighted P1D theory.
    window_matrix (np.ndarray): Real-valued Mxn window matrix, where N is FFT grid length
    model (np.ndarray): Real-valued vector length n of the original unbinned theory model
    If the window matrix is rectangular, this will also rebin the model
    '''
    return np.matmul(window_matrix, model)

def rebin_theta(V_aM_in_Theta, P_aM_in_Theta):
    '''
    Pass in an array of the P_aM within this coarse Theta bin
    as well as the weights V_aM within this bin
    Returns: average P estimate in Theta
    '''
    V_AM = np.sum(V_aM_in_Theta)
    P_AM = np.sum(V_aM_in_Theta * P_aM_in_Theta, axis=0) # this assumes simple top-hat binning
    return P_AM/V_AM