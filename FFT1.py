import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

def mi_funcion_sen( vmax, dc, ff, ph, nn, fs): 


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
_, yy = mi_funcion_sen(1, 0, ff, 0, N, fs)
#el _, me ignora tt, porque no lo necesito
FFT=fft(yy)
absFFT=np.abs(FFT)
#angleFFT=np.angle(FFT)

plt.stem(freqs, absFFT, linefmt="orchid", markerfmt="o", basefmt="orchid", label="N/4")

#-----------------(N/4 + 1)*df-----------------#
ff1=(N/4 + 1)*df
_, yy1 = mi_funcion_sen(1, 0, ff1, 0, N, fs)
FFT1=fft(yy1)
absFFT1=np.abs(FFT1)
#angleFFT1=np.angle(FFT1)
plt.stem(freqs, absFFT1, linefmt="lightseagreen", markerfmt="o", basefmt="lightseagreen", label="(N/4 + 1)df")      


#-----------------del medio-----------------#
ff2=(ff+ff1)/2
_, yy2 = mi_funcion_sen(1, 0, ff2, 0, N, fs)
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
tt, x = mi_funcion_sen(np.sqrt(2), 0, ff, 0, N, fs)
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


