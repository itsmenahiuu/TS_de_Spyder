import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft


def sen(vmax, dc, ff, ph, nn, fs):
    Ts = 1/fs 
    tt = np.linspace(0, (nn-1)*Ts, nn)
    xx = vmax*np.sin(2*np.pi*ff*tt + ph) + dc
    return tt, xx 


N = 1000
fs = N
df = fs/N
freqs = np.fft.fftfreq(N, 1/fs) #eje de frecuencias de la DFT (tiene la parte positiva y negativa, alineado con la salida de fft)

def calcularFFT(x):
    X = fft(x)/N                
    ABS = np.abs(X)          
    return ABS, freqs


#-----------------N/4-----------------#
ff0 = (N/4)*df
tt, x0 = sen(np.sqrt(2), 0, ff0, 0, N, fs)
ABS0, freqs = calcularFFT(x0)

#-----------------(N/4+0.25)*df-----------------#
ff1 = (N/4+0.25)*df
tt, x1 = sen(np.sqrt(2), 0, ff1, 0, N, fs)
ABS1, _ = calcularFFT(x1)

#-----------------(N/4+0.5)*df-----------------#
ff2 = (N/4+0.5)*df
tt, x2 = sen(np.sqrt(2), 0, ff2, 0, N, fs)
ABS2, _ = calcularFFT(x2)


varianza = np.var(x0)
print("Varianza de la señal:", varianza)

#-----------------Graficoss-----------------# 
plt.figure(1)
plt.plot(freqs, 10*np.log10(ABS0**2), "o", label="k0 = N/4", color = "mediumvioletred")
plt.plot(freqs, 10*np.log10(ABS2**2), "x", label="k0 = N/4+0.5", color = "darkmagenta")
plt.plot(freqs, 10*np.log10(ABS1**2), ".", label="k0 = N/4+0.25", color = "lightblue")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("[dB]")
plt.title("Densidad espectral de potencia")
plt.xlim([0, fs/2])
plt.grid()
plt.legend()
plt.show()

# %% Parseval

potenciaTiempo0 = np.sum(x0**2) 
potenciaTiempo1 = np.sum(x1**2) 
potenciaTiempo2 = np.sum(x2**2)  

FFTx0 = fft(x0)
FFTx1 = fft(x1)
FFTx2 = fft(x2)
potenciaFrecuencia0 = (1/N) * np.sum(np.abs(FFTx0)**2)
potenciaFrecuencia1 = (1/N) * np.sum(np.abs(FFTx1)**2)
potenciaFrecuencia2 = (1/N) * np.sum(np.abs(FFTx2)**2)

print(f"Potencia en tiempo para N/4: {potenciaTiempo0:.0f}")
print(f"Potencia en frecuencia para N/4: {potenciaFrecuencia0:.0f}")

print(f"Potencia en tiempo para N/4+0.25: {potenciaTiempo1:.0f}")
print(f"Potencia en frecuencia para N/4+0.25: {potenciaFrecuencia1:.0f}")

print(f"Potencia en tiempo para N/4+0.5: {potenciaTiempo2:.0f}")
print(f"Potencia en frecuencia para N/4+0.5: {potenciaFrecuencia1:.0f}")

# %%
#zero padding --------> mejora la resolucion espectral, interpolo
ffPadded= np.arange( 9*N) * (fs/(9*N))

#----------------N/4----------------#
zeroPadding0 = np.zeros(9*N)
zeroPadding0[:N] = x0
FFTpadding0=fft(zeroPadding0)
absFFTpadded0=np.abs(FFTpadding0)

#----------------N/4+0.25----------------#
zeroPadding1 = np.zeros(9*N)
zeroPadding1[:N] = x1
FFTpadding1=fft(zeroPadding1)
absFFTpadded1=np.abs(FFTpadding1)

#----------------N/4+0.5----------------#
zeroPadding2 = np.zeros(9*N)
zeroPadding2[:N] = x2
FFTpadding2=fft(zeroPadding2)
absFFTpadded2=np.abs(FFTpadding2)


plt.figure(2)
plt.plot(ffPadded, np.log10(absFFTpadded2**2)*10,"o", color="mediumpurple")
plt.plot(ffPadded, np.log10(absFFTpadded0**2)*10,"x", color="mediumturquoise")
plt.plot(ffPadded, np.log10(absFFTpadded1**2)*10,".", color="hotpink")

plt.title("Zero Padding")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("dB")
plt.grid(True)
plt.xlim([0, fs/2])





















