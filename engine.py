"""
Wi-Fi CSI MIMO Radar Simulation Engine
========================================
This module provides the core Digital Signal Processing (DSP) and radar algorithms
for simulating human target detection through walls using Channel State Information (CSI).

Key Features:
- MUSIC (Multiple Signal Classification) algorithm for Angle of Arrival (AoA) estimation
- Time of Flight (ToF) range estimation
- Micro-Doppler analysis for vital signs extraction
- Multipath fading simulation
- Optimized NumPy operations for high performance

Author: Senior Radar Systems Engineer & Principal Python Architect
Version: 1.0.0
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal
from numpy.typing import NDArray


class RadarEngine:
    """
    Core radar signal processing engine for Wi-Fi CSI-based MIMO radar simulation.

    This class implements advanced radar algorithms including MUSIC spectrum analysis
    for spatial localization and micro-Doppler signature extraction for vital signs.

    Attributes:
        num_antennas (int): Number of antenna elements in the MIMO array
        wavelength (float): Signal wavelength in meters (Wi-Fi 5GHz = 0.06m)
        antenna_spacing (float): Distance between antenna elements in meters
        range_resolution (float): Radar range resolution in meters
        angle_resolution (float): Angular resolution in degrees
    """

    def __init__(
        self,
        num_antennas: int = 8,
        frequency_ghz: float = 5.0,
        antenna_spacing_lambda: float = 0.5
    ) -> None:
        """
        Initialize the radar engine with system parameters.

        Args:
            num_antennas: Number of antenna elements (default: 8)
            frequency_ghz: Operating frequency in GHz (default: 5.0 for Wi-Fi)
            antenna_spacing_lambda: Antenna spacing in wavelengths (default: 0.5λ)
        """
        self.num_antennas = num_antennas
        self.frequency_ghz = frequency_ghz
        self.wavelength = 3e8 / (frequency_ghz * 1e9)  # c/f in meters
        self.antenna_spacing = antenna_spacing_lambda * self.wavelength
        self.range_resolution = 0.1  # meters
        self.angle_resolution = 1.0  # degrees

    def _compute_steering_vector(
        self,
        angle_deg: float
    ) -> NDArray[np.complex128]:
        """
        Compute the array steering vector for a given angle.

        The steering vector represents the phase relationship across antenna elements
        for a signal arriving from a specific direction.

        Args:
            angle_deg: Angle of arrival in degrees (-90 to 90)

        Returns:
            Complex steering vector of shape (num_antennas,)
        """
        angle_rad = np.deg2rad(angle_deg)
        # Phase shift between adjacent elements
        phase_shift = (2 * np.pi * self.antenna_spacing / self.wavelength) * np.sin(angle_rad)
        # Antenna element indices
        antenna_indices = np.arange(self.num_antennas)
        # Steering vector: a(θ) = [1, e^(jψ), e^(j2ψ), ..., e^(j(M-1)ψ)]
        steering_vector = np.exp(1j * phase_shift * antenna_indices)
        return steering_vector

    def _generate_signal_covariance(
        self,
        target_range: float,
        target_angle: float,
        snr_db: float,
        multipath: bool = False
    ) -> NDArray[np.complex128]:
        """
        Generate the signal covariance matrix with optional multipath effects.

        Args:
            target_range: Target distance in meters
            target_angle: Target angle in degrees
            snr_db: Signal-to-Noise Ratio in decibels
            multipath: Enable multipath fading simulation

        Returns:
            Covariance matrix of shape (num_antennas, num_antennas)
        """
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10.0)

        # Primary target steering vector
        a_primary = self._compute_steering_vector(target_angle)

        # Initialize covariance with primary target
        signal_power = snr_linear
        R_signal = signal_power * np.outer(a_primary, np.conj(a_primary))

        # Add multipath reflections (ghost targets)
        if multipath:
            # First multipath: wall reflection with -6dB attenuation
            ghost_angle_1 = target_angle + np.random.uniform(-15, -5)
            ghost_power_1 = signal_power * 0.25  # -6dB
            a_ghost_1 = self._compute_steering_vector(ghost_angle_1)
            R_signal += ghost_power_1 * np.outer(a_ghost_1, np.conj(a_ghost_1))

            # Second multipath: floor/ceiling reflection with -12dB attenuation
            ghost_angle_2 = target_angle + np.random.uniform(5, 15)
            ghost_power_2 = signal_power * 0.063  # -12dB
            a_ghost_2 = self._compute_steering_vector(ghost_angle_2)
            R_signal += ghost_power_2 * np.outer(a_ghost_2, np.conj(a_ghost_2))

        # Add noise covariance (identity matrix for white Gaussian noise)
        R_noise = np.eye(self.num_antennas, dtype=np.complex128)

        # Total covariance matrix
        R_total = R_signal + R_noise

        return R_total

    def advanced_music_spectrum(
        self,
        target_r: float,
        target_a: float,
        snr_db: float,
        multipath: bool = False,
        range_span: Tuple[float, float] = (0.5, 10.0),
        angle_span: Tuple[float, float] = (-60.0, 60.0),
        range_points: int = 100,
        angle_points: int = 121
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute the 2D MUSIC spatial spectrum using optimized NumPy broadcasting.

        This method implements the MUSIC (Multiple Signal Classification) algorithm
        for high-resolution Direction of Arrival (DoA) estimation. It uses eigenvalue
        decomposition to separate signal and noise subspaces.

        Optimization: Uses meshgrid and broadcasting to avoid nested loops,
        achieving O(1) Python loop overhead for massive performance gains.

        Args:
            target_r: True target range in meters
            target_a: True target angle in degrees
            snr_db: Signal-to-Noise Ratio in decibels
            multipath: Enable multipath fading effects
            range_span: (min_range, max_range) tuple in meters
            angle_span: (min_angle, max_angle) tuple in degrees
            range_points: Number of range grid points
            angle_points: Number of angle grid points

        Returns:
            Tuple of (ranges_1d, angles_1d, spectrum_2d_db):
                - ranges_1d: 1D array of range values
                - angles_1d: 1D array of angle values
                - spectrum_2d_db: 2D MUSIC spectrum in dB, shape (range_points, angle_points)
        """
        # Generate covariance matrix
        R = self._generate_signal_covariance(target_r, target_a, snr_db, multipath)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Determine number of signal sources (targets)
        num_sources = 3 if multipath else 1

        # Noise subspace eigenvectors
        E_n = eigenvectors[:, num_sources:]

        # Create 1D arrays for range and angle
        ranges_1d = np.linspace(range_span[0], range_span[1], range_points)
        angles_1d = np.linspace(angle_span[0], angle_span[1], angle_points)

        # **CRITICAL OPTIMIZATION: Use meshgrid and broadcasting**
        # Create 2D grids for vectorized computation
        angles_2d = np.linspace(angle_span[0], angle_span[1], angle_points)

        # Precompute all steering vectors for all angles (vectorized)
        # Shape: (num_antennas, angle_points)
        angles_rad = np.deg2rad(angles_2d)
        phase_shifts = (2 * np.pi * self.antenna_spacing / self.wavelength) * np.sin(angles_rad)
        antenna_indices = np.arange(self.num_antennas)[:, np.newaxis]  # (num_antennas, 1)

        # Broadcasting: (num_antennas, 1) * (1, angle_points) = (num_antennas, angle_points)
        all_steering_vectors = np.exp(1j * phase_shifts[np.newaxis, :] * antenna_indices)

        # Compute MUSIC pseudo-spectrum for all angles simultaneously
        # E_n shape: (num_antennas, num_noise_eigenvectors)
        # all_steering_vectors shape: (num_antennas, angle_points)

        # Project steering vectors onto noise subspace: a^H * E_n * E_n^H * a
        # Using optimized matrix multiplication
        E_n_conj_T = np.conj(E_n.T)  # (num_noise_eigenvectors, num_antennas)

        # Compute E_n^H * all_steering_vectors -> (num_noise_eigenvectors, angle_points)
        projection = E_n_conj_T @ all_steering_vectors

        # Compute ||E_n^H * a||^2 for each angle
        projection_norm_sq = np.sum(np.abs(projection) ** 2, axis=0)  # (angle_points,)

        # MUSIC pseudo-spectrum: P(θ) = 1 / (a^H * E_n * E_n^H * a)
        music_spectrum_angle = 1.0 / (projection_norm_sq + 1e-10)

        # Range-dependent attenuation (simplified model)
        # In real CSI, range is estimated from ToF/phase differences
        # Here we simulate range dependency with Gaussian centered on true target
        range_profile = np.exp(-((ranges_1d - target_r) ** 2) / (2 * 0.5 ** 2))

        # Broadcast to create 2D spectrum: (range_points, angle_points)
        spectrum_2d = range_profile[:, np.newaxis] * music_spectrum_angle[np.newaxis, :]

        # Convert to dB scale and normalize
        spectrum_2d_db = 10 * np.log10(spectrum_2d + 1e-10)
        spectrum_2d_db = spectrum_2d_db - np.max(spectrum_2d_db)  # Normalize to 0 dB peak
        spectrum_2d_db = np.clip(spectrum_2d_db, -40, 0)  # Clip at -40 dB noise floor

        return ranges_1d, angles_1d, spectrum_2d_db

    def extract_vital_signs(
        self,
        is_moving: bool,
        duration: float = 10.0,
        fs: float = 100.0
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Extract vital signs from micro-Doppler signature simulation.

        This method simulates the phase variations in CSI caused by chest wall
        movements during breathing and heartbeat for static targets, or chaotic
        body motion for moving targets.

        Args:
            is_moving: Target movement state (False=static, True=moving)
            duration: Signal duration in seconds
            fs: Sampling frequency in Hz

        Returns:
            Tuple of (time_array, amplitude_array):
                - time_array: Time samples in seconds, shape (num_samples,)
                - amplitude_array: Normalized amplitude signal, shape (num_samples,)
        """
        num_samples = int(duration * fs)
        time_array = np.linspace(0, duration, num_samples)

        if is_moving:
            # Moving target: Chaotic broadband noise (walking, limb motion)
            # Simulate random body reflections with varying Doppler shifts

            # Base walking frequency ~2 Hz with harmonics
            walk_freq = 2.0
            amplitude_array = np.zeros(num_samples)

            # Add multiple harmonics to simulate complex limb motion
            for harmonic in range(1, 6):
                freq = walk_freq * harmonic
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = 1.0 / harmonic  # Decreasing amplitude for higher harmonics
                amplitude_array += amplitude * np.sin(2 * np.pi * freq * time_array + phase)

            # Add broadband noise for chaotic motion
            noise_amplitude = 0.5
            amplitude_array += noise_amplitude * np.random.randn(num_samples)

            # Normalize
            amplitude_array = amplitude_array / (np.max(np.abs(amplitude_array)) + 1e-10)

        else:
            # Static target: Clean vital signs (breathing + heartbeat)

            # Breathing rate: ~20 breaths/min = 0.33 Hz
            breathing_freq = 0.33
            breathing_amplitude = 1.0
            breathing_signal = breathing_amplitude * np.sin(2 * np.pi * breathing_freq * time_array)

            # Heartbeat rate: ~72 bpm = 1.2 Hz
            heart_freq = 1.2
            heart_amplitude = 0.3  # Heartbeat is weaker than breathing
            heart_signal = heart_amplitude * np.sin(2 * np.pi * heart_freq * time_array)

            # Add slight physiological noise
            noise_amplitude = 0.05
            noise = noise_amplitude * np.random.randn(num_samples)

            # Combine signals
            amplitude_array = breathing_signal + heart_signal + noise

            # Normalize to [-1, 1] range
            amplitude_array = amplitude_array / (np.max(np.abs(amplitude_array)) + 1e-10)

        return time_array, amplitude_array

    def estimate_vital_rates(
        self,
        time_array: NDArray[np.float64],
        amplitude_array: NDArray[np.float64]
    ) -> Tuple[float, float]:
        """
        Estimate breathing and heart rates from vital signs signal using FFT.

        Args:
            time_array: Time samples in seconds
            amplitude_array: Amplitude signal

        Returns:
            Tuple of (breathing_rate_bpm, heart_rate_bpm)
        """
        # Compute sampling frequency
        fs = 1.0 / (time_array[1] - time_array[0])

        # Compute FFT
        fft_vals = np.fft.fft(amplitude_array)
        fft_freq = np.fft.fftfreq(len(amplitude_array), 1.0 / fs)

        # Only positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_magnitude = np.abs(fft_vals[positive_freq_idx])

        # Breathing range: 0.1 - 0.5 Hz (6 - 30 bpm)
        breathing_mask = (fft_freq >= 0.1) & (fft_freq <= 0.5)
        if np.any(breathing_mask):
            breathing_idx = np.argmax(fft_magnitude[breathing_mask])
            breathing_freq = fft_freq[breathing_mask][breathing_idx]
            breathing_rate_bpm = breathing_freq * 60.0
        else:
            breathing_rate_bpm = 0.0

        # Heart rate range: 0.8 - 2.0 Hz (48 - 120 bpm)
        heart_mask = (fft_freq >= 0.8) & (fft_freq <= 2.0)
        if np.any(heart_mask):
            heart_idx = np.argmax(fft_magnitude[heart_mask])
            heart_freq = fft_freq[heart_mask][heart_idx]
            heart_rate_bpm = heart_freq * 60.0
        else:
            heart_rate_bpm = 0.0

        return breathing_rate_bpm, heart_rate_bpm


# Module-level convenience function for quick testing
def create_default_engine() -> RadarEngine:
    """
    Create a radar engine with default Wi-Fi 5GHz parameters.

    Returns:
        Configured RadarEngine instance
    """
    return RadarEngine(num_antennas=8, frequency_ghz=5.0, antenna_spacing_lambda=0.5)
