import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, hilbert, resample
from scipy.interpolate import interp1d

class AdvancedVoiceAnonymizer:
    def __init__(self):
        self.speed_factor = 1.1
        self.formant_shift_factor = 1.02
        self.rate_factor = 1.0

    def design_bandpass_filter(self, lowcut, highcut, fs, order=5):
        """
        Design a bandpass filter with frequency validation
        """
        nyquist = 0.5 * fs

        # Ensure frequencies are within valid range
        lowcut = min(max(lowcut, 0), nyquist)
        highcut = min(max(highcut, 0), nyquist)

        # Normalize frequencies
        low = lowcut / nyquist
        high = highcut / nyquist

        # Ensure low < high and both are within (0, 1)
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))

        b, a = butter(order, [low, high], btype='band')
        return b, a

    def simulate_vocal_tract(self, signal, sr):
        """
        Simulate different vocal tract by changing playback speed
        """
        new_length = int(len(signal) / self.speed_factor)
        return resample(signal, new_length)

    def shift_formants(self, signal, sr):
        """
        Shift formants using bandpass filters
        """
        max_freq = sr / 2.1

        formant_ranges = [
            (200, min(800, max_freq)),
            (600, min(1600, max_freq)),
            (1400, min(2600, max_freq)),
            (2400, min(3500, max_freq)),
            (3300, min(4500, max_freq))
        ]

        modified_signal = np.zeros_like(signal, dtype=float)

        for low, high in formant_ranges:
            if high > low and high < sr/2:
                b, a = self.design_bandpass_filter(low, high, sr)
                filtered = filtfilt(b, a, signal)

                new_low = min(low * self.formant_shift_factor, sr/2.1)
                new_high = min(high * self.formant_shift_factor, sr/2.1)

                if new_high > new_low:
                    b_new, a_new = self.design_bandpass_filter(new_low, new_high, sr)
                    shifted = filtfilt(b_new, a_new, filtered)
                    modified_signal += shifted

        return modified_signal

    def estimate_speaking_rate(self, signal, sr):
        """
        Estimate speaking rate by detecting syllables through energy
        """
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)

        energy = []
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            energy.append(np.sum(frame**2))

        energy = np.array(energy)

        from scipy.signal import find_peaks
        peaks, _ = find_peaks(energy, distance=int(0.150 * sr / hop_length))

        return len(peaks) / (len(signal) / sr)

    def modify_speaking_rate(self, signal, sr):
        """
        Modify speaking rate using PSOLA
        """
        analytic_signal = hilbert(signal)
        phase = np.unwrap(np.angle(analytic_signal))
        pitch_marks = np.where(np.diff(phase) < 0)[0]

        new_pitch_marks = np.round(pitch_marks * self.rate_factor).astype(int)

        window_size = int(0.025 * sr)
        output = np.zeros(int(len(signal) * self.rate_factor))

        for i, mark in enumerate(new_pitch_marks):
            if mark < len(output) - window_size:
                if i < len(pitch_marks):
                    orig_mark = pitch_marks[i]
                    if orig_mark < len(signal) - window_size:
                        window = signal[orig_mark:orig_mark + window_size]
                        window = window * np.hanning(len(window))
                        output[mark:mark + window_size] += window

        return output

    def exchange_formant_bands(self, signal, sr):
        """
        Exchange F4 and F5 bands
        """
        max_freq = sr / 2.1

        f4_range = (2400, min(3500, max_freq))
        f5_range = (3300, min(4500, max_freq))

        if f4_range[1] > f4_range[0] and f5_range[1] > f5_range[0]:
            b_f4, a_f4 = self.design_bandpass_filter(f4_range[0], f4_range[1], sr)
            b_f5, a_f5 = self.design_bandpass_filter(f5_range[0], f5_range[1], sr)

            f4_band = filtfilt(b_f4, a_f4, signal)
            f5_band = filtfilt(b_f5, a_f5, signal)

            remaining_signal = signal - f4_band - f5_band
            return remaining_signal + f5_band + f4_band

        return signal

    def replace_high_bands_with_noise(self, signal, sr):
        """
        Replace F5-F9 bands with modulated pink noise
        """
        max_freq = sr / 2.1
        cutoff = min(3500, max_freq)

        # Generate pink noise
        noise_length = len(signal)
        pink_noise = np.random.normal(0, 1, noise_length)
        freqs = np.fft.fftfreq(len(pink_noise))
        f = np.abs(freqs)
        f[0] = 1e-6
        pink_filter = 1/np.sqrt(f)
        pink_noise_filtered = np.real(np.fft.ifft(np.fft.fft(pink_noise) * pink_filter))

        # Extract envelope
        b, a = self.design_bandpass_filter(cutoff, sr/2.2, sr)
        high_band = filtfilt(b, a, signal)
        envelope = np.abs(hilbert(high_band))

        # Modulate noise
        modulated_noise = pink_noise_filtered * envelope

        # Combine
        b, a = self.design_bandpass_filter(0, cutoff, sr)
        low_band = filtfilt(b, a, signal)

        return low_band + 0.3 * modulated_noise

def anonymize(input_audio_path):
    """
    anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`.
    sr : int
        The sample rate of the processed audio.
    """
    # Read the source audio file
    audio, sr = sf.read(input_audio_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Normalize input
    audio = audio / np.max(np.abs(audio))

    # Initialize anonymizer
    anonymizer = AdvancedVoiceAnonymizer()

    # Apply the complete anonymization pipeline
    # 1. Simulate different vocal tract
    audio = anonymizer.simulate_vocal_tract(audio, sr)

    # 2. Shift formants
    audio = anonymizer.shift_formants(audio, sr)

    # 3. Modify speaking rate
    audio = anonymizer.modify_speaking_rate(audio, sr)

    # 4. Exchange formant bands
    audio = anonymizer.exchange_formant_bands(audio, sr)

    # 5. Replace high bands with noise
    #audio = anonymizer.replace_high_bands_with_noise(audio, sr)

    # Final normalization
    audio = audio / np.max(np.abs(audio))

    return audio.astype(np.float32), sr