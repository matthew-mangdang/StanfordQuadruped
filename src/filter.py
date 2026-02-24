import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi
from scipy.fft import rfft, rfftfreq


class signal_processor:
    def __init__(self, signal=None, dt=None, filename=None):
        """
        Initialize the signal processor with a signal and sampling interval.
        Input: signal and sampling time
        """
        if signal is None:
            self.signal = None
            self.dt = dt
            self.fs = 1.0 / dt  # Sampling frequency
        else:
            self.signal = np.array(signal)
            self.dt = dt
            self.fs = 1.0 / dt  # Sampling frequency
            print(f"Initialized signal processor with {len(self.signal)} samples and sampling frequency {self.fs} Hz.")

        self.filtered_signal = None
        self.residual = None
        
        # Real-time filter state
        self.b = None
        self.a = None
        self.zi = None


    def fft(self):
        """
        Perform frequency domain analysis of the signal for analysis.
        """
        n = len(self.signal)
        freqs = rfftfreq(n, d=self.dt)
        fft_vals = np.abs(rfft(self.signal))
        
        return freqs, fft_vals
    
    def butter_filter(self, cutoff_freq, order=4):
        """
        Apply a low-pass Butterworth filter and store the result.
        """
        nyquist = 0.5 * self.fs
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        self.filtered_signal = filtfilt(b, a, self.signal)
        return self.filtered_signal



    def compute_residual(self):
        """
        Compute residual (original - filtered) to highlight high-frequency spikes.
        """
        if self.filtered_signal is None:
            raise ValueError("filtered_signal is not set. Run butter_filter() first.")
        self.residual = self.signal - self.filtered_signal
        return self.residual
    
    def plot_signals(self, data, filename):
        """
        General to plot the original, filtered, and residual signals for comparison.
        """

        time_vals = np.arange(len(data)) * self.dt 
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(time_vals, self.signal, label='Original Signal')
        axes[0].set_title('Original Signal')
        axes[0].set_ylabel('Amplitude') 

        axes[1].plot(time_vals, self.filtered_signal, label='Filtered Signal')
        axes[1].set_title('Filtered Signal')
        axes[1].set_ylabel('Amplitude')

        axes[2].plot(time_vals, self.residual, label='Residual Signal')
        axes[2].set_title('Residual Signal')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')

        plt.tight_layout()
        if filename:
            fig.savefig(filename, dpi=100)
        #plt.show()

    def plot_fft(self, filename=None):
        """
        Plot the signal and its FFT for visualization.
        """
        freqs, fft_vals = self.fft()
        time_vals = np.arange(len(self.signal)) * self.dt
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        axes[0].plot(time_vals, self.signal)
        axes[0].set_title('Signal (Time Domain)')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)

        axes[1].plot(freqs, fft_vals)
        axes[1].set_title('FFT of the Signal')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Magnitude')
        axes[1].grid(True)

        fig.tight_layout()
        if filename:
            fig.savefig(filename, dpi=100)
        #plt.show()
        return

    def load_test_signal(self, filename):
        """
        Load a test signal from a file for processing.
        """
        self.signal = np.load(filename)
        print(f"Loaded test signal from {filename} and sampling frequency {self.fs} Hz.")


class butter_filter_live():
    def __init__(self, cutoff_freq, dt, order=4):
        """
        Set up a real-time Butterworth filter for continuous sample processing.
        The optimal settings can be found using the another object "signal_processor".
        
        Input:
            cutoff_freq: Cutoff frequency in Hz
            dt: Sampling period in seconds (i.e., 1/fs)
            order: Filter order (default 4)
        """
        self.dt = dt
        self.fs = 1.0 / dt  # Sampling frequency
        nyquist = 0.5 * self.fs
        normalized_cutoff = cutoff_freq / nyquist
        print(f"Initialized signal processor and sampling frequency {self.fs} Hz.")
        self.b, self.a = butter(order, normalized_cutoff, btype='low', analog=False)
        self.zi = lfilter_zi(self.b, self.a) * 0.0  # Initialize with zero input

    def apply_filter(self, new_sample):
        """
        Use initialized filter coefficients to process a new sample in real-time.
        """
        filtered_sample, self.zi = lfilter(self.b, self.a, [new_sample], zi=self.zi)
        return filtered_sample[0]

