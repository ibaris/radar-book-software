# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 24.01.2023 by bari_is*
*Copyright (C) 2023*
*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
from typing import Literal, Union
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.constants import pi as PI
from scipy.fftpack import next_fast_len
from scipy.signal.windows import hann, hamming
from scipy.interpolate import interp1d
from scipy.fftpack import ifft, fftshift
import warnings
from scipy.special import binom
from scipy.signal.windows import chebwin, hann, hamming, blackmanharris, kaiser

# %matplotlib qt

WINDOW_TYPE = Literal["Uniform", "Binomial", "Tschebyscheff",
                      "Kaiser", "Blackman-Harris", "Hanning", "Hamming"]


# ----------------------------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------------------------
def antenna_pattern_coefficients(number_of_elements: int,
                                 window_type: WINDOW_TYPE = "Uniform",
                                 side_lobe_level: float = 0) -> np.ndarray:
    """Compute the coefficients for the antenna pattern.

    Parameters
    ----------
    number_of_elements : int
        The number of elements in the array.
    window_type : WINDOW_TYPE, optional
        The string name of the window., by default "Uniform"
    side_lobe_level : float, optional
        The sidelobe level for Tschebyscheff window in [dB], by default 0

    Returns
    -------
    np.ndarray

    Raises
    ------
    AssertionError
        The window type is not supported. 
    """
    # Calculate the coefficients
    if window_type == 'Uniform':
        coefficients = np.ones(number_of_elements)
    elif window_type == 'Binomial':
        coefficients = binom(number_of_elements-1, range(0, number_of_elements))
    elif window_type == 'Tschebyscheff':
        warnings.simplefilter("ignore", UserWarning)
        coefficients = chebwin(number_of_elements, at=side_lobe_level, sym=True)
    elif window_type == 'Kaiser':
        coefficients = kaiser(number_of_elements, 6, True)
    elif window_type == 'Blackman-Harris':
        coefficients = blackmanharris(number_of_elements, True)
    elif window_type == 'Hanning':
        coefficients = hann(number_of_elements, True)
    elif window_type == 'Hamming':
        coefficients = hamming(number_of_elements, True)

    else:
        raise AssertionError(
            f"Window type {window_type} not understood. Available window types are {WINDOW_TYPE}.")

    return coefficients


def antenna_factor(coefficients: np.ndarray,
                   scan_angle: float,
                   element_spacing: float,
                   frequency: float,
                   theta: float,
                   offset: int,
                   odd_case: bool) -> float:
    """Calculate the array factor for a linear binomial excited array.

    Parameters
    ----------
    coefficients : np.ndarray
        The antenna pattern coefficients. See `antenna_pattern_coefficients`.
    scan_angle : float
        The angle to which the main beam is scanned in [rad].
    element_spacing : float
        The distance between elements.
    frequency : float
        he operating frequency in [Hz].
    theta : float
        The angle at which to evaluate the array factor [rad].
    offset : int
        Offset for even/odd case. This is the number of elements divided by 2. 
    odd_case : bool
        Determine if the number of elements is odd or even.

    Returns
    -------
    float
    """
    # Calculate the wavenumber
    k = 2.0 * PI * frequency / SPEED_OF_LIGHT

    # Calculate the phase
    psi = k * element_spacing * (np.cos(theta) - np.cos(scan_angle))

    # Odd case
    if odd_case:
        coefficients = np.roll(coefficients, offset + 1)
        coefficients[0] *= 0.5
        return sum(coefficients[i] * np.cos(i * psi) for i in range(offset + 1))
    # Even case
    else:
        coefficients = np.roll(coefficients, offset)
        return sum(coefficients[i] * np.cos((i + 0.5) * psi) for i in range(offset))


def back_projection(signal: np.ndarray,
                    sensor_position: np.ndarray,
                    range_center: Union[float, int, list, np.ndarray],
                    image: np.ndarray,
                    frequency_step: float,
                    frequency_start: float,
                    fft_length: int) -> np.ndarray:
    """
    Reconstruct the two-dimensional image using the filtered backprojection method.

    Parameters
    ----------
    signal : np.ndarray
        The signal in K-space.
    sensor_position : np.ndarray
        An array with the sensor [x, y, z] coordinates in [m].
    range_center : Union[float, int, list, np.ndarray]
        The range to the center of the image in [m].
    image : np.ndarray
        The [x, y, z] coordinates of the image in [m].
    frequency_step : float
        This ist the first frequency minus the last frequency of the frequency array.
    frequency_start : float
        The first frequency in the frequency array.
    fft_length : int
        The number of points in the FFT.

    Returns
    -------
    np.ndarray
    """
    # Calculate the maximum scene size and resolution
    range_extent = SPEED_OF_LIGHT / (2.0 * frequency_step)

    # Calculate the range window for the pulses
    range_window = np.linspace(-0.5 * range_extent, 0.5 * range_extent, fft_length)

    # Initialize the image
    bp_image = np.zeros_like(image[0], dtype=complex)

    # Loop over all pulses in the data
    # term = 1j * 4.0 * PI * frequency[0] / SPEED_OF_LIGHT
    term = 1j * 4.0 * PI * frequency_start / SPEED_OF_LIGHT

    # To work with stripmap
    if not isinstance(range_center, list):
        range_center *= np.ones(len(sensor_position[0]))

    index = 0
    for xs, ys, zs in zip(sensor_position[0], sensor_position[1], sensor_position[2]):

        # Calculate the range profile
        range_profile = fftshift(ifft(signal[:, index], fft_length))

        # Create the interpolation for this pulse
        f = interp1d(range_window, range_profile, kind='linear', bounds_error=False, fill_value=0.0)

        # Calculate the range to each pixel
        range_image = np.sqrt((xs - image[0]) ** 2 + (ys - image[1]) ** 2 +
                              (zs - image[2]) ** 2) - range_center[index]

        # Interpolate the range profile onto the image grid and multiply by the range phase
        # For large scenes, should check the range window and index
        bp_image += f(range_image) * np.exp(term * range_image)

        index += 1

    return bp_image


# ----------------------------------------------------------------------------------------------
# Define Frame
# ----------------------------------------------------------------------------------------------
squint_angle = np.radians(0)

# * Set the **x** and **y** image center
x_center = 1000
y_center = x_center * np.tan(squint_angle)

# * Set the range to the center of the image (m), the **x** location of the target, they **y** location of the target, the target RCS (m^2)
#! These are only point targets.
range_center = 1e3
xt = [-3.0, 8.0]
yt = [10.0, -40.0]
rt = [5.0, 10.0]

# * Set image span in the **x** and **y** directions (This is the size of the image measured from the center).
x_span = 30.0
y_span = 100.0

# * Set the number of bins in the **x** and **y** directions
nx = 400
ny = 400

# * Alternatively, define the resolution of the system and calculate the bins automatically.
x_resolution = 0.25
y_resolution = 0.5

# nx = int(np.floor(x_span / x_resolution))
# ny = int(np.floor(y_span / y_resolution))

# Create Image Space ===============================================================
# * Set up the image space (m)
xi = np.linspace(-0.5 * x_span + x_center, 0.5 * x_span + x_center, nx)
yi = np.linspace(-0.5 * y_span + y_center, 0.5 * y_span + y_center, ny)

x_image, y_image = np.meshgrid(xi, yi)
z_image = np.zeros_like(x_image)

image = np.array([x_image, y_image, z_image])

# ----------------------------------------------------------------------------------------------
# Sensor Parameter
# ----------------------------------------------------------------------------------------------
# * Set the aperture length (m) and the antenna width (m)
aperture_length = 100
antenna_width = 2.0

# * Set the start freuqency (Hz) and the bandwidth (Hz) of the transmitted signal
start_frequency = 1e9
bandwidth = 100e6

# * Calculate the wavelength at the start frequency (m)
wavelength = SPEED_OF_LIGHT / start_frequency

# Setup Frequency Space ============================================================
# * Calculate the number of frequencies
df = SPEED_OF_LIGHT / (2.0 * np.sqrt(x_span ** 2 + y_span ** 2))
number_of_frequencies = int(np.ceil(bandwidth / df))

# * Set up the frequency space
frequency = np.linspace(start_frequency, start_frequency + bandwidth, number_of_frequencies)
frequency_step = frequency[1] - frequency[0]
frequency_start = frequency[0]

# * Set the length of the FFT
fft_length = next_fast_len(4 * number_of_frequencies)

# Setup Sub-Aperture ===============================================================
# * Calculate the element spacing (m)
element_spacing = wavelength / 4.0

# * Calculate the number of antenna elements
number_of_elements = int(np.ceil(antenna_width / element_spacing + 1))
odd_case = number_of_elements & 1
offset = int(np.floor(number_of_elements / 2))

# * Calculate the spacing on the synthetic aperture (m)
aperture_spacing = np.tan(SPEED_OF_LIGHT / (2 * y_span * start_frequency)) * x_center  # Based on y_span

# * Calculate the number of samples (pulses) on the aperture
number_of_samples = int(np.ceil(aperture_length / aperture_spacing + 1))

# * Create the aperture
synthetic_aperture = np.linspace(-0.5 * aperture_length, 0.5 * aperture_length, number_of_samples)

# Sensor Position ==================================================================
# * Calculate the sensor location
sensor_x = np.zeros_like(synthetic_aperture)
sensor_y = synthetic_aperture
sensor_z = np.zeros_like(synthetic_aperture)

sensor_position = np.array([sensor_x, sensor_y, sensor_z])

# ----------------------------------------------------------------------------------------------
# Signal Calculation
# ----------------------------------------------------------------------------------------------
# * Initialize the signal
signal = np.zeros([number_of_frequencies, number_of_samples], dtype=complex)

# * Initialize the range center (m)
range_center = np.zeros_like(synthetic_aperture)

# * Phase term for the range phase (rad)
phase_term = -1j * 4.0 * PI * frequency / SPEED_OF_LIGHT

# * Initialize the antenna coefficients
antenna_coef = antenna_pattern_coefficients(number_of_elements=number_of_elements,
                                            window_type="Uniform",
                                            side_lobe_level=0)

# * Create the signal (k-space)
for index, sa in enumerate(synthetic_aperture):

    range_center[index] = np.sqrt(x_center ** 2 + (y_center - sa) ** 2)

    for x, y, r in zip(xt, yt, rt):

        # Antenna pattern at each target
        target_range = np.sqrt((x_center + x) ** 2 + (y_center + y - sa) ** 2) - range_center[index]

        target_azimuth = np.arctan((y_center + y - sa) / (x_center + x))

        antenna_pattern = antenna_factor(coefficients=antenna_coef,
                                         scan_angle=0.5 * np.pi - squint_angle,
                                         element_spacing=element_spacing,
                                         frequency=start_frequency,
                                         theta=0.5 * np.pi - target_azimuth,
                                         offset=offset,
                                         odd_case=odd_case) * np.cos(squint_angle)

        signal[:, index] += r * antenna_pattern ** 2 * np.exp(phase_term * target_range)

# ----------------------------------------------------------------------------------------------
# Filter Signal
# ----------------------------------------------------------------------------------------------
# * Set the window type (Rectangular, Hanning, or Hamming)
window_type = 'Hanning'

# * Get the selected window
if window_type == 'Hanning':
    h1 = hann(number_of_frequencies, True)
    h2 = hann(number_of_samples, True)
    coefficients = np.sqrt(np.outer(h1, h2))

elif window_type == 'Hamming':
    h1 = hamming(number_of_frequencies, True)
    h2 = hamming(number_of_samples, True)
    coefficients = np.sqrt(np.outer(h1, h2))

elif window_type == 'Rectangular':
    coefficients = np.ones([number_of_frequencies, number_of_samples])

signal_filtered = signal * coefficients

# ----------------------------------------------------------------------------------------------
# Backprojection of Image
# ----------------------------------------------------------------------------------------------
# * Reconstruct the image using the `backprojection` routines
bp_image = back_projection(signal=signal,
                           sensor_position=sensor_position,
                           range_center=range_center,
                           image=image,
                           frequency_step=frequency_step,
                           frequency_start=frequency_start,
                           fft_length=fft_length)

# Normalize the image
bpi = np.abs(bp_image) / np.amax(abs(bp_image))
bpi_db = 20.0 * np.log10(bpi)

# Plot Image =======================================================================
# * Set the dynamic range for the image (dB)
dynamic_range = 50

fig, axes1 = plt.subplots(figsize=(6, 9))

# create the color plot
im = axes1.pcolor(xi, yi, bpi_db, cmap='jet', vmin=-dynamic_range, vmax=0, shading='auto')
cbar = fig.colorbar(im, ax=axes1, orientation='vertical')
cbar.set_label("(dB)", size=10)

# Set the plot title and labels
axes1.set_title('Back Projected Signal', size=14)
axes1.set_xlabel('Range [m]', size=12)
axes1.set_ylabel('Azimuth [m]', size=12)

# Plot Signal ======================================================================
# signal_i = abs(signal) / np.amax(abs(signal))  # ! Normalize the Image.
# signal_idb = 20.0 * np.log10(signal_i)

# fig, axes1 = plt.subplots()

# # create the color plot
# im = axes1.imshow(signal_idb)
# cbar = fig.colorbar(im, ax=axes1, orientation='horizontal')
# cbar.set_label("Decibel", size=10)

# # Set the plot title and labels
# axes1.set_title('Unfiltered Signal', size=10)
# axes1.set_ylabel('Frequency', size=10)
# axes1.set_xlabel('Azimuth in [m]', size=10)

# Plot Filtered Signal =============================================================
# signal_filtered_i = abs(signal_filtered) / np.amax(abs(signal_filtered))  # ! Normalize the Image.
# signal_filtered_idb = 20.0 * np.log10(signal_filtered_i)

# fig, axes1 = plt.subplots()

# # create the color plot
# im = axes1.imshow(signal_filtered_idb)
# cbar = fig.colorbar(im, ax=axes1, orientation='horizontal')
# cbar.set_label("Decibel", size=10)

# # Set the plot title and labels
# axes1.set_title('Filtered Signal', size=10)
# axes1.set_ylabel('Frequency', size=10)
# axes1.set_xlabel('Azimuth in [m]', size=10)
