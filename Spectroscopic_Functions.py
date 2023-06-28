import numpy as np
from globals_and_params import *
# Useful Spectroscopic Functions

# Psuedo Voigts can create lorentz and Guassian functions or a convolution of both
# It is useful for background creation


def Gaussian(x, amp, center, width):
    try:
        return amp * np.e**(-((x - center)**2. / (2. * width**2.)))
    except ZeroDivisionError:
        width += 0.0001
        Gaussian(x, amp, center, width)


def Lorentzian(x, amp, center, width):
    try:
        return amp * ((width**2) / ((x - center)**2 + width**2))
    except ZeroDivisionError:
        width += 0.0001
        Lorentzian(x, amp, center, width)


def Voigt(x, amp, center, width, shape):
    return shape * Lorentzian(x,
                              amp,
                              center,
                              width) + (1. - shape) * Gaussian(x,
                                                               amp,
                                                               center,
                                                               width)


def Abs_Lorentz(x, amp, center, width):
    try:
        return amp / (np.pi) * (width / ((x - center)**2 + width**2))
    except ZeroDivisionError:
        width += 0.0001
        Abs_Lorentz(x, amp, center, width)


def graphite(E, alpha, y0, b):
    # This was calculated with wavelength as the X axis, so
    # wavelen is the X array in wavelength instead of energy
    wavelen = WAVELENGTH_TO_ENERGY_CONVERSION / E

    norm_profile = alpha * (y0 + np.exp(-b * wavelen))
    return norm_profile


def Onsager_diff(epsilon, eta):
    """
    Difference between Solvent's Onsager Polarity Functions
    Term for Adjusting the Excitonic Energy to Environmental Effects
        Units: Dimensionless
        Terms:
            epsilon := Dielectric Constant of the Solvent
            eta := Refractive Index of the Solvent
        Reference 4, 5
    """
    fe = Onsager(epsilon)
    fn = Onsager(eta**2)
    return fe - fn


def Onsager(x):
    """
    Onsager Polarity Function
    Used for Adjusting the Excitonic Energy to Environmental Effects
        Units: Dimensionless
        Reference 4, 5
    """

    return 2 * (x - 1) / (2 * x + 1)
