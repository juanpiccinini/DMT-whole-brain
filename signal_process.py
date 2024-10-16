import numpy as np
from scipy.fftpack import fftfreq, fft, ifft, fft2
import matplotlib.pyplot as plt


def signal_detrend(t_series):
    """It detrends the signal by subtracting the mean and
    then normalazing it divinding by the standar deviation

    The input should be an (1 x time) array"""
    
    mean = np.mean(t_series)
    std = np.std(t_series)
    signal = (t_series - mean)/std


    return signal


def fft_2d(matrix):
    # Take the fourier transform of the image.
    F1 = np.fft.fft2(matrix)
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = np.fft.fftshift( F1 )
    # Calculate a 2D power spectrum
    psd2D = np.abs( F2 )

    return psd2D

  
