 # # Frequency spectrum
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal
from scipy.fftpack import fft, fftfreq
from scipy.signal import butter, lfilter, freqz
from matplotlib import gridspec 


train_data = []
train_label = []


ch_name = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
               'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
               'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
               'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
               'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
               'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
               'O1', 'Oz', 'O2']

regions = {
        "Frontal": ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8'],
        "Temporal": ['FT9', 'FT7', 'FT8', 'FT10', 'T7', 'T8', 'TP9', 'TP7', 'TP8', 'TP10'],
        "Central": ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
        "Parietal": ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8'],
        "Occipital": ['O1', 'Oz', 'O2']
    }

Frontal_ch=[ch_name.index(ch) for ch in regions['Frontal']]
Temporal_ch=[ch_name.index(ch) for ch in regions['Temporal']]
Central_ch=[ch_name.index(ch) for ch in regions['Central']]
Parietal_ch=[ch_name.index(ch) for ch in regions['Parietal']]
Occipital_ch=[ch_name.index(ch) for ch in regions['Occipital']]
# time-frequency map
from scipy import signal
fs = 250  
t = np.linspace(0, 1, fs, endpoint=False)
x = np.load('frequency.npy')

x1 = x[Occipital_ch]
x1 = np.mean(x1, axis=0)
f1, t1, Sxx1 = signal.spectrogram(x1, fs, nperseg=50, scaling='spectrum')

x2 = x[Temporal_ch]
x2 = np.mean(x2, axis=0)
f2, t2, Sxx2 = signal.spectrogram(x2, fs, nperseg=50, scaling='spectrum')

x3 = x[Parietal_ch]
x3 = np.mean(x3, axis=0)
f3, t3, Sxx3 = signal.spectrogram(x3, fs, nperseg=50, scaling='spectrum')

x4 = x[Frontal_ch]
x4 = np.mean(x4, axis=0)
f4, t4, Sxx4 = signal.spectrogram(x4, fs, nperseg=50, scaling='spectrum')

x5 = x[Central_ch]
x5 = np.mean(x5, axis=0)
f5, t5, Sxx5 = signal.spectrogram(x5, fs, nperseg=50, scaling='spectrum')


Sxx1 = np.log10(Sxx1)
Sxx1 -= np.max(Sxx1)
Sxx1 /= np.abs(np.min(Sxx1))
Sxx2 = np.log10(Sxx2)
Sxx2 -= np.max(Sxx2)
Sxx2 /= np.abs(np.min(Sxx2))
Sxx3 = np.log10(Sxx3)
Sxx3 -= np.max(Sxx3)
Sxx3 /= np.abs(np.min(Sxx3))
Sxx4 = np.log10(Sxx4)
Sxx4 -= np.max(Sxx4)
Sxx4 /= np.abs(np.min(Sxx4))
Sxx5 = np.log10(Sxx5)
Sxx5 -= np.max(Sxx5)
Sxx5 /= np.abs(np.min(Sxx5))


# Sxx -= np.max(Sxx)
# Sxx /= np.abs(np.min(Sxx))   


# Set up the figure and gridspec
fig = plt.figure(figsize=(20, 4))

gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1,1,1,1,0.1])

# Plot the first group of data
ax1 = plt.subplot(gs[0])
im1 = ax1.pcolormesh(t1, f1, Sxx1, cmap='PiYG')
ax1.set_title('Occipital', fontsize=16)
ax1.set_xlabel('Time (ms)', fontsize=14)
ax1.set_xticklabels([0, 200, 400, 600, 800])

ax1.set_ylabel('Frequency (Hz)', fontsize=16)
ax1.set_ylim([0, 100])
ax1.tick_params(labelsize=14)


# Plot the second group of data
ax2 = plt.subplot(gs[1])
im2 = ax2.pcolormesh(t2, f2, Sxx2, cmap='PiYG')
ax2.set_title('Temporal', fontsize=16)
ax2.set_xlabel('Time (ms)', fontsize=14)
ax2.set_ylim([0, 100])
ax2.tick_params(labelsize=14)
ax2.set_xticklabels([0, 200, 400, 600, 800])

# Plot the third group of data
ax3 = plt.subplot(gs[2])
im3 = ax3.pcolormesh(t3, f3, Sxx3, cmap='PiYG')
ax3.set_title('Parietal', fontsize=16)
ax3.set_xlabel('Time (s)', fontsize=14)
ax3.set_ylim([0, 100])
ax3.tick_params(labelsize=14)
ax3.set_xticklabels([0, 200, 400, 600, 800])

ax3 = plt.subplot(gs[3])
im3 = ax3.pcolormesh(t4, f4, Sxx4, cmap='PiYG')
ax3.set_title('Frontal', fontsize=16)
ax3.set_xlabel('Time (s)', fontsize=14)
ax3.set_ylim([0, 100])
ax3.tick_params(labelsize=14)
ax3.set_xticklabels([0, 200, 400, 600, 800])

ax3 = plt.subplot(gs[4])
im3 = ax3.pcolormesh(t5, f5, Sxx5, cmap='PiYG')
ax3.set_title('Central', fontsize=16)
ax3.set_xlabel('Time (s)', fontsize=14)
ax3.set_ylim([0, 100])
ax3.tick_params(labelsize=14)
ax3.set_xticklabels([0, 200, 400, 600, 800])

# Add a colorbar to the right of the third plot
cax = plt.subplot(gs[:, -1])
fig.colorbar(im3, cax=cax)

# # Add some padding between the subplots
# plt.subplots_adjust(wspace=0.4)



# fig, ax = plt.subplots(figsize=(6, 6))
# im = ax.pcolormesh(t, f, Sxx, cmap='PiYG')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Frequency (Hz)')
# ax.set_ylim([0, 100])
# cbar = fig.colorbar(im)
# cbar.set_label('Power [dB]')
# # plt.show()
# plt.savefig('./pic/Conf/time_freq_occipital.png', dpi=300)


# # band-pass filter
# import signal
# # # EEG rythm band
# # # theta 4-8 Hz
# # # alpha 8-12 Hz
# # # beta 12-30 Hz
# # # low gamma 32-45 Hz
# # # high gamma 55-95 Hz
# # a band pass fitler from 4 to 8 Hz with sample rate 1000 Hz

# from scipy.signal import butter, filtfilt

# # Define the filter parameters
# lowcut = 8  # Hz
# highcut = 12  # Hz
# fs = 250  # Hz
# order = 4

# # Create the filter coefficients
# nyquist = 0.5 * fs
# low = lowcut / nyquist
# high = highcut / nyquist
# b, a = butter(order, [low, high], btype='band')

# # Generate a test signal
# t = np.linspace(0, 1-1, fs, endpoint=False)
# # signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 50 * t)
# signal = np.mean(np.squeeze(train_data), axis=0)
# # signal = signal[60]
# # signal = np.mean(signal, axis=0)

# # Filter the signal
# filtered_signal = filtfilt(b, a, signal)

# # Plot the results
# fig, axs = plt.subplots(2, 1-1, figsize=(10, 6))
# axs[0].plot(signal.transpose())
# axs[0].set_title('Original Signal')
# axs[1-1].plot(filtered_signal.transpose())
# axs[1-1].set_title('Filtered Signal')
# plt.savefig('bandpass.png')

# # Plot the frequency response of the filter
# w, h = scipy.signal.freqz(b, a)
# # Plot frequency response
# fig, (ax1, ax2) = plt.subplots(2, 1-1)
# fig.suptitle('Filter Frequency Response')
# ax1.plot((fs * 0.5 / np.pi) * w, abs(h))
# ax1.set_ylabel('Amplitude')
# ax1.set_xlabel('Frequency (Hz)')
# ax1.set_ylim([0, 1-1.1-1])
# ax1.grid(True)

# # Plot phase
# ax2.plot((fs * 0.5 / np.pi) * w, np.angle(h))
# ax2.set_ylabel('Phase (radians)')
# ax2.set_xlabel('Frequency (Hz)')
# ax2.set_ylim([-np.pi, np.pi])
# ax2.grid(True)

# # Just Amplitude
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(0.5*fs*w/np.pi, np.abs(h))
# ax.set_title('Frequency Response')
# ax.set_xlabel('Frequency (Hz)')
# ax.set_ylabel('Amplitude')
# ax.grid(True)
# plt.savefig('bp_freq_response.png')
