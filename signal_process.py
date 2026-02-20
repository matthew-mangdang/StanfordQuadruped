from src.filter import signal_processor

signal_handler = signal_processor(signal=None,dt=0.01) 
signal_handler.load_test_signal(filename="acceleration_trotting_hit2.npy")
signal_handler.plot_fft(filename="./result/acceleration_trotting_hit2_fft.png")
signal_handler.butter_filter(cutoff_freq=15.0, order=1)
signal_handler.compute_residual()
signal_handler.plot_signals(data=signal_handler.signal, filename="./result/ax_trotting_hit2_10hz.png")
