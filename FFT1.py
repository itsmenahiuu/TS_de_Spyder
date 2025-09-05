import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

def sen( vmax, dc, ff, ph, nn, fs): 


    Ts = 1/fs 
    tt = np.linspace(0, (nn-1)*Ts, nn) 

 
    xx=vmax*np.sin(2*np.pi*ff*tt+ph)+dc 

    return tt,xx 

N=1000 ## cuanto mas chico el N, mas separados se ven los palitos de  diferentes frecuencias
fs=N #
df=fs/N # resolución espectral
#En una FFT, el eje x no son muestras de tiempo sino índices de frecuencia
#como grafico de 0 a N/2, mis frecuecias quedan de fs/2
freqs = np.arange(0, N) * df
plt.figure(1)

#-----------------N/4-----------------#
ff=(N/4)*df
_, yy = sen(1, 0, ff, 0, N, fs)
#el _, me ignora tt, porque no lo necesito
FFT=fft(yy)
absFFT=np.abs(FFT)
#angleFFT=np.angle(FFT)

plt.stem(freqs, absFFT, linefmt="orchid", markerfmt="o", basefmt="orchid", label="N/4")

#-----------------(N/4 + 1)*df-----------------#
ff1=(N/4 + 1)*df
_, yy1 = sen(1, 0, ff1, 0, N, fs)
FFT1=fft(yy1)
absFFT1=np.abs(FFT1)
#angleFFT1=np.angle(FFT1)
plt.stem(freqs, absFFT1, linefmt="lightseagreen", markerfmt="o", basefmt="lightseagreen", label="(N/4 + 1)df")      


#-----------------del medio-----------------#
ff2=(ff+ff1)/2
_, yy2 = sen(1, 0, ff2, 0, N, fs)
FFT2=fft(yy2)
absFFT2=np.abs(FFT2)
#angleFFT2=np.angle(FFT2)
plt.stem(freqs, absFFT2, linefmt="deepskyblue", markerfmt="o", basefmt="deepskyblue", label="medio")


plt.title("FFT de senoidales")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X[k]|")
plt.grid(True)
plt.xlim(0, N/2)
plt.legend()
#Si la frecuencia de la señal es múltiplo exacto de df, la FFT te da un solo palito sino, no





#--------------------------GRAFICO EN dB--------------------------#

plt.figure(2)
#-----------------N/4-----------------#
plt.plot(freqs, np.log10(absFFT)*20,"o", label="N/4", color="indigo")

#-----------------(N/4 + 1)*df-----------------#
plt.plot(freqs, np.log10(absFFT1)*20,"x", label="N/4 + 1", color="mediumvioletred")

#-----------------del medio-----------------#
plt.plot(freqs, np.log10(absFFT2)*20,"x", label="N/4 + 0,5", color="lightseagreen")


plt.title("FFT de senoidales")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("dB")
plt.grid(True)
plt.xlim([0, fs/2])
plt.legend()


# %% Actividad de Parseval

#pomerle a=raiz(2) para que la varianza me de 1 (es normalizar)
tt, x = sen(np.sqrt(2), 0, ff, 0, N, fs)
FFTp=fft(x)
absFFTp=np.abs(FFTp)
varianza = np.var(x)
print("Varianza de la señal:", varianza)

#densidad espectral de potencia 
potenciaEspectral = absFFTp**2
dBpotencia = 10 * np.log10(potenciaEspectral)

plt.figure(3)
plt.plot(freqs, dBpotencia,"x", color="mediumvioletred")
plt.title("Espectro en Potencia [dB]")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("10 log10(|X[k]|^2)")
plt.grid(True)
plt.xlim([0, fs/2])

#verifico parseval
energiaENtiempo = np.sum(x**2)
energiaENfrecuencia = (1/N) * np.sum(absFFTp**2)

print("Energía en el tiempo:", energiaENtiempo)
print("Energía en la frecuencia:", energiaENfrecuencia)


# %%
#zero padding --------> mejora la resolucion espectral, interpolo

#----------------N/4 (con potencia unitaria, oseaa A=raiz(2))----------------#
zeroPadding = np.zeros(10*N)
zeroPadding[:N] = x
FFTpadding=fft(zeroPadding)
ffPadded= np.arange( 10*N) * (fs/(10*N))
absFFTpadded=np.abs(FFTpadding)

plt.figure(4)
plt.plot(ffPadded, np.log10(absFFTpadded**2)*10,"o", color="mediumturquoise")
plt.plot(freqs, np.log10(absFFTp**2)*10, "x", color="mediumvioletred")
plt.title("Zero Padding")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("dB")
plt.grid(True)
plt.xlim([0, fs/2])

# %%
N = 64   # longitud de la ventana
freqsVentanas = np.arange(0, N) * df

#Mis ventanitas
flattop = np.flattop(N)
hamming = np.hamming(N)
blackmanHarris = np.blackmanharris(N)

# FFT
FFTflattop = fft(flattop)
FFThamming = fft(hamming)
FFTblackmanHarris = fft(blackmanHarris)



plt.figure(6)
plt.plot(freqsVentanas, FFThamming, label="Rectangular")

plt.title("Espectro de las ventanas")
plt.xlabel("ω [rad/muestra]")
plt.ylabel("|W(ω)| [dB]")

plt.grid(True)
plt.legend()
np
plt.show()

#hacer lo de SNR

# %%
# Parámetros
N = 64   # longitud de la ventana
freqsVentanas = np.arange(0, N) * df

#Mis ventanitas
flattop = np.flattop(N)
hamming = np.hamming(N)
blackmanHarris = np.blackmanharris(N)

# FFT
FFTflattop = fft(flattop)
FFThamming = fft(hamming)
FFTblackmanHarris = fft(blackmanHarris)

# Eje de frecuencia en radianes (de -pi a pi)
omega = np.linspace(-np.pi, np.pi, Nfft)

# Normalización y paso a dB
def to_dB(W):
    return 20*np.log10(np.abs(np.fft.fftshift(W)) / np.max(np.abs(W)))

# ----------------- VENTANAS EN TIEMPO ----------------- #
plt.figure(4)
plt.plot(w_rect, "b", label="Rectangular")
plt.plot(w_hamming, "g", label="Hamming")
plt.plot(w_hann, color="orange", label="Hann")
plt.plot(w_blackman, "r", label="Blackman")
plt.title("Ventanas en el tiempo")
plt.xlabel("n")
plt.ylabel("w[n]")
plt.grid(True)
plt.legend()

# ----------------- ESPECTRO EN FRECUENCIA ----------------- #
plt.figure(5)
plt.plot(omega, to_dB(Wrect), "b", label="Rectangular")
plt.plot(omega, to_dB(Whamming), "g", label="Hamming")
plt.plot(omega, to_dB(Whann), color="orange", label="Hann")
plt.plot(omega, to_dB(Wblackman), "r", label="Blackman")
plt.title("Espectro de las ventanas")
plt.xlabel("ω [rad/muestra]")
plt.ylabel("|W(ω)| [dB]")
plt.ylim([-80, 5])
plt.grid(True)
plt.legend()

plt.show()













