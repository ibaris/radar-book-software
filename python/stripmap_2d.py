# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 24.01.2023 by bari_is*
*Copyright (C) 2023*
*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
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

%matplotlib qt


# ----------------------------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------------------------
def array_factor(number_of_elements, scan_angle, element_spacing, frequency, theta, window_type, side_lobe_level):
    """
    Calculate the array factor for a linear binomial excited array.
    :param window_type: The string name of the window.
    :param side_lobe_level: The sidelobe level for Tschebyscheff window (dB).
    :param number_of_elements: The number of elements in the array.
    :param scan_angle: The angle to which the main beam is scanned (rad).
    :param element_spacing: The distance between elements.
    :param frequency: The operating frequency (Hz).
    :param theta: The angle at which to evaluate the array factor (rad).
    :return: The array factor as a function of angle.
    """
    # Calculate the wavenumber
    k = 2.0 * PI * frequency / SPEED_OF_LIGHT

    # Calculate the phase
    psi = k * element_spacing * (np.cos(theta) - np.cos(scan_angle))

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

    # Calculate the offset for even/odd
    offset = int(np.floor(number_of_elements / 2))

    # Odd case
    if number_of_elements & 1:
        coefficients = np.roll(coefficients, offset + 1)
        coefficients[0] *= 0.5
        return sum(coefficients[i] * np.cos(i * psi) for i in range(offset + 1))
    # Even case
    else:
        coefficients = np.roll(coefficients, offset)
        return sum(coefficients[i] * np.cos((i + 0.5) * psi) for i in range(offset))


def reconstruct(signal, sensor_x, sensor_y, sensor_z, range_center, x_image, y_image, z_image, frequency, fft_length):
    """
    Reconstruct the two-dimensional image using the filtered backprojection method.
    :param signal: The signal in K-space.
    :param sensor_x: The sensor x-coordinate (m).
    :param sensor_y: The sensor y-coordinate (m).
    :param sensor_z: The sensor z-coordinate (m).
    :param range_center: The range to the center of the image (m).
    :param x_image: The x-coordinates of the image (m).
    :param y_image: The y-coordinates of the image (m).
    :param z_image: The z-coordinates of the image (m).
    :param frequency: The frequency array (Hz).
    :param fft_length: The number of points in the FFT.
    :return: The reconstructed image.
    """
    # Get the frequency step size
    frequency_step = frequency[1] - frequency[0]

    # Calculate the maximum scene size and resolution
    range_extent = SPEED_OF_LIGHT / (2.0 * frequency_step)

    # Calculate the range window for the pulses
    range_window = np.linspace(-0.5 * range_extent, 0.5 * range_extent, fft_length)

    # Initialize the image
    bp_image = np.zeros_like(x_image, dtype=complex)

    # Loop over all pulses in the data
    term = 1j * 4.0 * PI * frequency[0] / SPEED_OF_LIGHT

    # To work with stripmap
    if not isinstance(range_center, list):
        range_center *= np.ones(len(sensor_x))

    index = 0
    for xs, ys, zs in zip(sensor_x, sensor_y, sensor_z):

        # Calculate the range profile
        range_profile = fftshift(ifft(signal[:, index], fft_length))

        # Create the interpolation for this pulse
        f = interp1d(range_window, range_profile, kind='linear', bounds_error=False, fill_value=0.0)

        # Calculate the range to each pixel
        range_image = np.sqrt((xs - x_image) ** 2 + (ys - y_image) ** 2 +
                              (zs - z_image) ** 2) - range_center[index]

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
range_center = 1e3
xt = [-3.0, 8.0]
yt = [10.0, -40.0]
rt = [5.0, 10.0]

# * Set image span in the **x** and **y** directions
x_span = 30.0
y_span = 100.0

# * Set the number of bins in the **x** and **y** directions
nx = 400
ny = 400

# Create Image Space ===============================================================
# * Set up the image space (m)
xi = np.linspace(-0.5 * x_span + x_center, 0.5 * x_span + x_center, nx)
yi = np.linspace(-0.5 * y_span + y_center, 0.5 * y_span + y_center, ny)

x_image, y_image = np.meshgrid(xi, yi)
z_image = np.zeros_like(x_image)

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

# * Set the length of the FFT
fft_length = next_fast_len(4 * number_of_frequencies)

# Setup Sub-Aperture ===============================================================
# * Calculate the element spacing (m)
element_spacing = wavelength / 4.0

# * Calculate the number of antenna elements
number_of_elements = int(np.ceil(antenna_width / element_spacing + 1))

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

# ----------------------------------------------------------------------------------------------
# Signal Calculation
# ----------------------------------------------------------------------------------------------
# * Initialize the signal
signal = np.zeros([number_of_frequencies, number_of_samples], dtype=complex)

# * Initialize the range center (m)
range_center = np.zeros_like(synthetic_aperture)

# * Phase term for the range phase (rad)
phase_term = -1j * 4.0 * PI * frequency / SPEED_OF_LIGHT

# * Create the signal (k-space)
for index, sa in enumerate(synthetic_aperture):

    range_center[index] = np.sqrt(x_center ** 2 + (y_center - sa) ** 2)

    for x, y, r in zip(xt, yt, rt):

        # Antenna pattern at each target
        target_range = np.sqrt((x_center + x) ** 2 + (y_center + y - sa) ** 2) - range_center[index]

        target_azimuth = np.arctan((y_center + y - sa) / (x_center + x))

        antenna_pattern = array_factor(number_of_elements,
                                       0.5 * np.pi - squint_angle,
                                       element_spacing,
                                       start_frequency,
                                       0.5 * np.pi - target_azimuth,
                                       'Uniform',
                                       0) * np.cos(squint_angle)

        signal[:, index] += r * antenna_pattern ** 2 * np.exp(phase_term * target_range)

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

# ----------------------------------------------------------------------------------------------
# Filter Signal
# ----------------------------------------------------------------------------------------------
# * Set the window type (Rectangular, Hanning, or Hamming)
window_type = 'Hamming'

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

# Plot Signal ======================================================================
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

# ----------------------------------------------------------------------------------------------
# Backprojection of Image
# ----------------------------------------------------------------------------------------------
# * Reconstruct the image using the `backprojection` routines
bp_image = reconstruct(signal, sensor_x, sensor_y, sensor_z, range_center,
                       x_image, y_image, z_image, frequency, fft_length)


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
