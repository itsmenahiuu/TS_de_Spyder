import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import wavfile

#--------------Para encontrar mi banda ancha--------------#
def estimar_BW(PSD, ff, cota):
    df = ff[1] - ff[0]
    energia_acumulada = np.cumsum(PSD * df)
    energia_total = energia_acumulada[-1]
    energia_corte = energia_total * cota
    idx_corte = np.where(energia_acumulada >= energia_corte)[0][0]
    frec_BW = ff[idx_corte]
    return frec_BW

eps = 1e-12

# %%
#--------------ECG--------------#
fs_ecg = 1000
ecg = np.load('ecg_sin_ruido.npy')
N_ecg = len(ecg)

plt.figure(1)
plt.plot(ecg, color='mediumvioletred')
plt.title('ECG sin ruido')
plt.grid()
plt.show()

nperseg = N_ecg // 30
window = 'blackman'
nfft = 6 * nperseg

# Welch
freqW, ecgW = sig.welch(ecg, fs=fs_ecg, window=window, nperseg=nperseg, nfft=nfft)
# Periodograma con ventaneado
freqP, ecgP = sig.periodogram(ecg, fs=fs_ecg, window=window, nfft=nfft)
# Blackman-Tukey
corr = sig.correlate(ecg - np.mean(ecg), ecg - np.mean(ecg), mode='full')/ len(ecg) 
#lo de -np.mea es para centrar la señal antes de estimar la autocorrelación
corr = corr[corr.size // 2:]  #la mitad positiva
win_bt = sig.windows.bartlett(len(corr))
corr_win = corr * win_bt
PSD_bt = np.abs(np.fft.rfft(corr_win, n=nfft))
freqBT = np.fft.rfftfreq(nfft, d=1/fs_ecg)

plt.figure(2)
plt.plot(freqW, 10*np.log10(ecgW + eps), label='Welch', color="cornflowerblue")
plt.plot(freqP, 10*np.log10(ecgP + eps), label='Periodograma', color="palevioletred")
plt.plot(freqBT, 10*np.log10(PSD_bt + eps), label='Blackman-Tukey', color="limegreen")
plt.title('ECG')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.xlim([0, 50])
plt.grid()
plt.legend()
plt.show()
# Ancho de banda y potencia total (Welch)
df = freqW[1] - freqW[0]
potTotal = np.sum(ecgW) * df
f95 = estimar_BW(ecgW, freqW, 0.95)
f97 = estimar_BW(ecgW, freqW, 0.97)
f99 = estimar_BW(ecgW, freqW, 0.99)

print("=====================================")
print("ECG")
print(f"Potencia total (Welch): {potTotal:.4e} V^2")
print(f"BW con 95%: {f95:.2f} Hz")
print(f"BW con 97%: {f97:.2f} Hz")
print(f"BW con 99%: {f99:.2f} Hz")
print("=====================================")



# %% 
#--------------PPG--------------#
fs_ppg = 400
ppg = np.load('ppg_sin_ruido.npy')
N_ppg = len(ppg)

plt.figure(3)
plt.plot(ppg, color='yellowgreen')
plt.title('PPG sin ruido')
plt.grid()
plt.show()

nperseg = N_ppg // 20
win = 'hann'
nfft = 2 * nperseg

freqW, ppgW = sig.welch(ppg, fs=fs_ppg, window=win, nperseg=nperseg, nfft=nfft)
freqP, ppgP = sig.periodogram(ppg, fs=fs_ppg, window=win, nfft=nfft)
corr = sig.correlate(ppg - np.mean(ppg), ppg - np.mean(ppg), mode='full')/ len(ecg)
corr = corr[corr.size // 2:]
win_bt = sig.windows.bartlett(len(corr))
corr_win = corr * win_bt
PSD_bt = np.abs(np.fft.rfft(corr_win, n=nfft))
freqBT = np.fft.rfftfreq(nfft, d=1/fs_ppg)

plt.figure(4)
plt.plot(freqW, 10*np.log10(ppgW + eps), label='Welch', color="cornflowerblue")
plt.plot(freqP, 10*np.log10(ppgP + eps), label='Periodograma', color="palevioletred")
plt.plot(freqBT, 10*np.log10(PSD_bt + eps), label='Blackman-Tukey', color="limegreen")
plt.title('PPG - Métodos de estimación de PSD (dB)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.xlim([0, 10])
plt.grid()
plt.legend()
plt.show()

df = freqW[1] - freqW[0]
potTotal = np.sum(ppgW) * df
f95 = estimar_BW(ppgW, freqW, 0.95)
f97 = estimar_BW(ppgW, freqW, 0.97)
f99 = estimar_BW(ppgW, freqW, 0.99)

print("=====================================")
print("PPG")
print(f"Potencia total (Welch): {potTotal:.4e} V^2")
print(f"BW con 95%: {f95:.2f} Hz")
print(f"BW con 97%: {f97:.2f} Hz")
print(f"BW con 99%: {f99:.2f} Hz")
print("=====================================")



# %%
#--------------AUDIO--------------#
fs_audio, audio = wavfile.read('la cucaracha.wav')
if audio.ndim > 1:
    audio = audio[:,0]
N_aud = len(audio)

plt.figure(5)
plt.plot(audio, color='darkorchid')
plt.title('Audio "la cucaracha"')
plt.grid()
plt.show()

nperseg = N_aud // 200
win = 'hann'
nfft = 2 * nperseg

freqW, audW = sig.welch(audio, fs=fs_audio, window=win, nperseg=nperseg, nfft=nfft)
freqP, audP = sig.periodogram(audio, fs=fs_audio, window=win, nfft=nfft)
corr = sig.correlate(audio - np.mean(audio), audio - np.mean(audio), mode='full')/ len(ecg)
corr = corr[corr.size // 2:]
win_bt = sig.windows.bartlett(len(corr))
corr_win = corr * win_bt
PSD_bt = np.abs(np.fft.rfft(corr_win, n=nfft))
freqBT = np.fft.rfftfreq(nfft, d=1/fs_audio)

plt.figure(6)
plt.plot(freqW, 10*np.log10(audW + eps), label='Welch', color="cornflowerblue")
plt.plot(freqP, 10*np.log10(audP + eps), label='Periodograma', color="palevioletred")
plt.plot(freqBT, 10*np.log10(PSD_bt + eps), label='Blackman-Tukey', color="limegreen")
plt.title('Audio - Métodos de estimación de PSD (dB)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.xlim([0, fs_audio/2])
plt.grid()
plt.legend()
plt.show()

df = freqW[1] - freqW[0]
potTotal = np.sum(audW) * df
f95 = estimar_BW(audW, freqW, 0.95)
f97 = estimar_BW(audW, freqW, 0.97)
f99 = estimar_BW(audW, freqW, 0.99)

print("=====================================")
print("AUDIO")
print(f"Potencia total (Welch): {potTotal:.4e} V^2")
print(f"BW con 95%: {f95:.2f} Hz")
print(f"BW con 97%: {f97:.2f} Hz")
print(f"BW con 99%: {f99:.2f} Hz")
print("=====================================")

# %%

#BONUS

##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_ruido = mat_struct['ecg_lead'].flatten()  # señal con ruido (1D)
N_ecg_ruido = len(ecg_ruido)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

plt.figure(7)
plt.plot(ecg_ruido, color='mediumvioletred')
plt.title('ECG con ruido')
plt.xlabel('Muestras [n]')
plt.ylabel('Amplitud [V]')
plt.grid()
plt.show()

# Parámetros (idénticos a los del ECG sin ruido)
nperseg = N_ecg_ruido // 30
window = 'blackman'
nfft = 6 * nperseg

# Welch
freqW_r, ecgW_r = sig.welch(ecg_ruido, fs=fs_ecg, window=window, nperseg=nperseg, nfft=nfft)

# Periodograma
freqP_r, ecgP_r = sig.periodogram(ecg_ruido, fs=fs_ecg, window=window, nfft=nfft)

# Blackman–Tukey
corr_r = sig.correlate(ecg_ruido - np.mean(ecg_ruido), ecg_ruido - np.mean(ecg_ruido), mode='full') / N_ecg_ruido
corr_r = corr_r[corr_r.size // 2:]
win_bt = sig.windows.bartlett(len(corr_r))
corr_win_r = corr_r * win_bt
PSD_bt_r = np.abs(np.fft.rfft(corr_win_r, n=nfft))
freqBT_r = np.fft.rfftfreq(nfft, d=1/fs_ecg)

# ---- Gráfico ----
plt.figure(8)
plt.plot(freqP_r, 10*np.log10(ecgP_r + eps), label='Periodograma', color="palevioletred")
plt.plot(freqBT_r, 10*np.log10(PSD_bt_r + eps), label='Blackman-Tukey', color="limegreen")
plt.plot(freqW_r, 10*np.log10(ecgW_r + eps), label='Welch', color="cornflowerblue")
plt.title('ECG con ruido')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB]')
plt.xlim([0, 50])
plt.grid()
plt.legend()
plt.show()

# ---- Potencia total y ancho de banda ----
df_r = freqW_r[1] - freqW_r[0]
potTotal_r = np.sum(ecgW_r) * df_r
f95_r = estimar_BW(ecgW_r, freqW_r, 0.95)
f97_r = estimar_BW(ecgW_r, freqW_r, 0.97)
f99_r = estimar_BW(ecgW_r, freqW_r, 0.99)

print("=====================================")
print("ECG CON RUIDO")
print(f"Potencia total (Welch): {potTotal_r:.4e} V^2")
print(f"BW con 95%: {f95_r:.2f} Hz")
print(f"BW con 97%: {f97_r:.2f} Hz")
print(f"BW con 99%: {f99_r:.2f} Hz")
print("=====================================")
























