import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io.wavfile import write


def estimar_BW(PSD, ff, cota = 0.98):
    energia_acumulada = np.cumsum(PSD) # El ultimo valor del vector contiene la suma de todos los anteriores (integral de toda la curva)
    energia_acumulada_normalizada = energia_acumulada / energia_acumulada[-1] # De tamaño (nperseg, 1)
    corte = energia_acumulada_normalizada[-1] * cota
    elementos_discriminados = int (np.where(energia_acumulada_normalizada >= corte)[0][0]) # cota es el porcentaje que determina los valores que me quiero quedar de la señal (ej.: cota = 0.99)
    frec_BW = ff[elementos_discriminados]
    return frec_BW


# %%


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


print("=====================================")
print(f"Potencia total: {potTotal:.4e} V^2")
print(f"BW con 95%: {frec95:.2f} Hz")
print(f"BW con 97%: {frec97:.2f} Hz")
print(f"BW con 99%: {frec99:.2f} Hz")
print("=====================================")



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
