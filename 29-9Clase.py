import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import write

##################
## ECG sin ruido
##################
fs_ecg = 1000  #frecuencia de muestreo en Hz
#Mi N es 30000

ecg_one_lead = np.load('ecg_sin_ruido.npy')

Necg = ecg_one_lead.shape[0]

plt.figure(1)
plt.plot(ecg_one_lead, color="mediumvioletred")
plt.title("ECG sin ruido")
plt.grid()
plt.show()

#welch

cantPromedio = 30  #Cantidad de promedios que quiero hacer N/nperseg, multiplo de 5 10 y par
#Para elegir la cantidad de promedio debo ver que no haya mucha varianza y que tmp se me deforme mucho el espectro
#me fijo con poca cantidad de promedio maso el pico, de ahi subo el promedio hasta que vea que tiene poca varianza y nada se movio
#como minimo mil muestras (nperseg)
nperseg = Necg//cantPromedio
print(f"nperseg usado en Welch: {nperseg}") 
window= 'blackman'
nfft = 6 * nperseg #como promedio poco ya tengo resoluciones importantes
freqW, ecgW = sig.welch(ecg_one_lead, fs=fs_ecg, window=window, nperseg=nperseg, nfft=nfft) 

#grafico la psd
plt.figure(2)
plt.plot(freqW, ecgW, color="mediumorchid")
plt.title("PSD del ECG sin ruido con Welch")
plt.xlim([0, 50])
plt.grid()
plt.show()


#calculo paso en frecuencia y potencia total
df = freqW[1] - freqW[0]                 #paso en frecuencia (Hz)
potTotal = np.sum(ecgW) * df            #potencia total (integral discreta)

#potencia acumulada con el cumsum
potAcum = np.cumsum(ecgW) * df          #integral acumulada desde 0 Hz

# frecuencia donde se alcanza el 95% de la potencia
nivel95 = 0.95 * potTotal
indice95 = np.where(potAcum >= nivel95)[0][0]   # primer índice donde se supera el 95%
frec95 = freqW[indice95]

#para 97%
nivel97 = 0.97 * potTotal
indice97 = np.where(potAcum >= nivel97)[0][0]
frec97 = freqW[indice97]

#para 99%
nivel99 = 0.99 * potTotal
indice99 = np.where(potAcum >= nivel99)[0][0]
frec99 = freqW[indice99]

print("=====================================")
print(f"Potencia total: {potTotal:.4e} V^2")
print(f"BW con 95%: {frec95:.2f} Hz")
print(f"BW con 97%: {frec97:.2f} Hz")
print(f"BW con 99%: {frec99:.2f} Hz")
print("=====================================")









# #periodogram
# freqP, ecgP = sig.periodogram(ecg_one_lead, fs=fs_ecg)

# #PSD perio
# plt.figure()
# plt.plot(freqP, ecgP, color="hotpink")
# plt.title("PSD del ECG sin ruido con Periodograma")
# plt.grid()
# plt.show()


# %%

####################################
# Lectura de pletismografía (PPG)  #
####################################

fs_ppg = 400 # Hz

ppg = np.load('ppg_sin_ruido.npy')

Nppg = ppg.shape[0]

plt.figure(1)
plt.plot(ppg)
plt.title("PPG sin ruido")
plt.grid()
plt.show()


cantPromedio = 20
nperseg = Nppg//cantPromedio
win = 'hann' 
nfft = 2 * nperseg  

freqW, ppgW = sig.welch(ppg, fs=fs_ppg, window=win, nperseg=nperseg, nfft=nfft)


plt.figure(2)
plt.plot(freqW, ppgW, color="mediumorchid")
plt.title("PSD del PPG con Welch")
plt.xlim([0, 50])
plt.grid()
plt.show()


# %%

####################
# Lectura de audio #
####################

# Cargar el archivo CSV como un array de NumPy
fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
Naud = wav_data.shape[0]

plt.figure(1)
plt.plot(wav_data)

cantPromedio = 200
nperseg = Naud//cantPromedio
win = 'hann' 
nfft = 2 * nperseg  

freqW, audW = sig.welch(ppg, fs=fs_ppg, window=win, nperseg=nperseg, nfft=nfft)


plt.figure(2)
plt.plot(freqW, audW, color="mediumorchid")
plt.title("PSD de la cucaracha con Welch")
plt.xlim([0, 50])
plt.grid()
plt.show()
