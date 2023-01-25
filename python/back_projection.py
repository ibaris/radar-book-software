# -*- coding: utf-8 -*-
"""
HEADER
======
*Created on 24.01.2023 by bari_is*
*Copyright (C) 2023*
*For COPYING and LICENSE details, please refer to the LICENSE file*

"""
import matplotlib.pyplot as plt
from numpy import zeros, dot, exp
import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.constants import pi as PI
from scipy.fftpack import next_fast_len
from scipy.signal.windows import hann, hamming
from scipy.interpolate import interp1d
from scipy.fftpack import ifft, fftshift


# ----------------------------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------------------------
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
        bp_image += f(range_image) * exp(term * range_image)

        index += 1

    return bp_image


# ----------------------------------------------------------------------------------------------
# Define Frame
# ----------------------------------------------------------------------------------------------
range_center = 1e3  # * Image Center Coordinate [m]
xt = [-3.0, 4.0]  # * X Coordinates of Objects (Relative to Image Center) [m]
yt = [1.0, -2.0]  # * Y Coordinates of Objects (Relative to Image Center) [m]
rt = [20.0, 10.0]  # * Radar Cross Section [m²]

# * Set image span in the x and y directions
x_span = 12.0
y_span = 12.0

# * Set the number of bins in the x and **y** directions
nx = 200
ny = 200

# Create Image Space ===============================================================
# * Create up the image space
xi = np.linspace(-0.5 * x_span, 0.5 * x_span, nx)  # * Spacing in X direction.
yi = np.linspace(-0.5 * y_span, 0.5 * y_span, ny)  # * Spacing in Y direction.

#! I dont understand this part. It generates three two dimensional images.
x_image, y_image = np.meshgrid(xi, yi)
z_image = np.zeros_like(x_image)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
np.meshgrid(a, b)

# ----------------------------------------------------------------------------------------------
# Sensor Parameter
# ----------------------------------------------------------------------------------------------
# * Set the start freuqency (Hz) and the bandwidth (Hz) of the transmitted signal
start_frequency = 9.65e+9
bandwidth = 1.5e+8

# Setup Azimuth Space ==============================================================
azimuth_start = -3.0  # * Start Azimuth Angle relative to center in [DEG]
azimuth_end = 3.0  # * Stop Azimuth Angle relative to center in [DEG]

image_length = np.sqrt(x_span ** 2 + y_span ** 2)

da = SPEED_OF_LIGHT / (2.0 * image_length * start_frequency)

na = int(np.radians(azimuth_end - azimuth_start) / da)  # * Amount of Azimuth steps.

azimuth_list = np.linspace(azimuth_start, azimuth_end, na)  # * Total Azimuth variation.

# Setup Frequency Space ============================================================
df = SPEED_OF_LIGHT / (2.0 * image_length)
nf = int(bandwidth / df)  # * Amount of Frequency steps.
frequencies = np.linspace(start_frequency, start_frequency + bandwidth, nf)  # * Total Frequency variation.

# * Set up the FFT length
fft_length = 8 * next_fast_len(nf)

# Sensor Position ==================================================================
# * Set up the aperture positions
sensor_x = range_center * np.cos(np.radians(azimuth_list))
sensor_y = range_center * np.sin(np.radians(azimuth_list))
sensor_z = np.zeros_like(sensor_x)

# ----------------------------------------------------------------------------------------------
# Signal Calculation
# ----------------------------------------------------------------------------------------------
signal = zeros([nf, na], dtype=complex)

for index, azimuth in enumerate(azimuth_list):

    sensor_los = [np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))]

    for x_obj, y_obj, rcs_obj in zip(xt, yt, rt):

        r_target = -dot(sensor_los, [x_obj, y_obj])

        signal[:, index] += rcs_obj * exp(-1j * 4.0 * PI * frequencies / SPEED_OF_LIGHT * r_target)

# Plot Signal ======================================================================
signal_filtered_i = abs(signal) / np.amax(abs(signal))  # ! Normalize the Image.
signal_filtered_idb = 20.0 * np.log10(signal_filtered_i)

fig, axes1 = plt.subplots()

# create the color plot
im = axes1.imshow(signal_filtered_idb)
cbar = fig.colorbar(im, ax=axes1, orientation='horizontal')
cbar.set_label("Decibel", size=10)

# Set the plot title and labels
axes1.set_title('Unfiltered Signal', size=10)
axes1.set_ylabel('Frequency', size=10)
axes1.set_xlabel('Azimuth in [m]', size=10)

# ----------------------------------------------------------------------------------------------
# Filter Signal
# ----------------------------------------------------------------------------------------------
window_type = 'Hanning'

if window_type == 'Hanning':
    h1 = hann(nf, True)
    h2 = hann(na, True)
    coefficients = np.sqrt(np.outer(h1, h2))

elif window_type == 'Hamming':
    h1 = hamming(nf, True)
    h2 = hamming(na, True)

    coefficients = np.sqrt(np.outer(h1, h2))

elif window_type == 'Rectangular':
    coefficients = np.ones([nf, na])

# * Apply the windowing coefficients
signal_filtered = signal * coefficients

# Plot Signal ======================================================================
signal_filtered_i = abs(signal_filtered) / np.amax(abs(signal_filtered))  # ! Normalize the Image.
signal_filtered_idb = 20.0 * np.log10(signal_filtered_i)

fig, axes1 = plt.subplots()

# create the color plot
im = axes1.imshow(signal_filtered_idb)
cbar = fig.colorbar(im, ax=axes1, orientation='horizontal')
cbar.set_label("Decibel", size=10)

# Set the plot title and labels
axes1.set_title('Filtered Signal', size=10)
axes1.set_ylabel('Frequency', size=10)
axes1.set_xlabel('Azimuth in [m]', size=10)

# ----------------------------------------------------------------------------------------------
# Backprojection of Image
# ----------------------------------------------------------------------------------------------
bp_image = reconstruct(signal_filtered, sensor_x, sensor_y, sensor_z, range_center,
                       x_image, y_image, z_image, frequencies, fft_length)

dynamic_range = 40  # * Set the dynamic range for the image (dB)

bpi = abs(bp_image) / np.amax(abs(bp_image))  # ! Normalize the Image.
bpi_db = 20.0 * np.log10(bpi)

# Plot Image =======================================================================
fig, axes1 = plt.subplots()

# create the color plot
im = axes1.pcolor(xi, yi, bpi_db, cmap='jet', vmin=-dynamic_range, vmax=0, shading='auto')
cbar = fig.colorbar(im, ax=axes1, orientation='vertical')
cbar.set_label("(dB)", size=10)

# Set the plot title and labels
axes1.set_title('Back Projected Signal', size=14)
axes1.set_xlabel('Range [m]', size=12)
axes1.set_ylabel('Azimuth [m]', size=12)

# Set the tick label size

axes1.tick_params(labelsize=12)
