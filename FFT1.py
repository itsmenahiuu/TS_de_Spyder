import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal as sig
win = sig.windows


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
Nbig=10000
Nv=31 #si ponia 1000 no se veia nada

#longitud de la ventana
freqsWin = np.linspace(-fs/2, fs/2, Nbig) 

#Mis ventanitas
flattop = win.flattop(Nv)
hamming = win.hamming(Nv)
blackmanHarris = win.blackmanharris(Nv)
rectangular = win.boxcar(Nv) 
gaussian = win.gaussian(Nv, std=0.4*Nv)

#agrego zero padding asi se ve mas lisito todo y se ve como en el holton
def ffPadding(x, Np, fs): #arme funcion que aplica zero padding a una señal x y calcula su FFT, asi no tengo que hacerlo 40 veces para cada ventana

#x: señal original
#N: cantidad total de puntos (incluye padding)
#fs: frecuencia de muestreo
    zeroPadding = np.zeros(Np)
    zeroPadding[:len(x)] = x 
    FFTpadding = fft(zeroPadding)
    absFFTpadded = np.abs(FFTpadding)
    ffPadded = np.arange(Np) * (fs / Np)
    return ffPadded, absFFTpadded
#ffPadded: eje de frecuencia [Hz]
#absFFTpadded: módulo de la FFT con padding

# FFT con padding
_, FFTflattop = ffPadding(flattop, Nbig, fs)
_, FFThamming = ffPadding(hamming, Nbig, fs)
_, FFTblackmanHarris = ffPadding(blackmanHarris, Nbig, fs)
_, FFTrectangular = ffPadding(rectangular, Nbig, fs)
_, FFTgaussian = ffPadding(gaussian, Nbig, fs)


#funcion para normalizar y pasar a dB (asi no lo hago 40 veces tmb)
def dB(W):
    return 20 * np.log10(np.abs(np.fft.fftshift(W)) / np.max(np.abs(W)))
#el shift me pone elcero en el centro, y el max hace que mi lobulo principal tenga su maximo en 0dB
#cada ventana tiene su propio valor medio (donde estal el maximo) por eso uso el shift

plt.figure(6)
plt.plot(freqsWin, dB(FFTgaussian), label="Gaussian", color="lightgreen")
plt.plot(freqsWin, dB(FFTflattop), label="Flattop", color="skyblue")
plt.plot(freqsWin, dB(FFThamming), label="Hamming", color="hotpink")
plt.plot(freqsWin, dB(FFTblackmanHarris), label="Blackman-Harris", color="mediumvioletred")
plt.plot(freqsWin, dB(FFTflattop), label="Flattop", color="purple")

plt.title("Respuesta en frecuencia de ventanas")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel(r"$|W(e^{j\omega})|$ [dB]") #Si, busque una forma elegante de decir "magnitud de la respuesta en frecuencia de la ventana"
plt.ylim([-60, 5])
plt.grid(True)
plt.legend()
plt.show()



# %%
#SNR

tt, señal = sen(1, 0, ff, 0, N, fs)

np.random.seed(0)  
ruido = np.random.normal(0, 1, N)# Ruido blanco gaussiano :O
#genero N muestras de una distribución normal con media=0, desviacion estandar=1 y ademas seed(0) me asegura que el ruido sea reproducible cada vez que corro mi codigo

señalConRuidito = señal + ruido

potenciaSenal = np.mean(señal**2)
potenciaRuido = np.mean(ruido**2)
SNRdB = 10 * np.log10(potenciaSenal/potenciaRuido)

print("SNR [dB]:", SNRdB)

FFTseñalConRuidito = fft(señalConRuidito)
absFFTseñalConRuidito = np.abs(FFTseñalConRuidito)

plt.figure(7)
plt.plot(freqs, 20 * np.log10(absFFT), label="Señal limpia", color="mediumvioletred")
plt.plot(freqs, 20 * np.log10(absFFTseñalConRuidito), label="Señal con ruido", color="deepskyblue")
plt.title("Comparación espectral de señal vs señal con ruido")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.legend()
plt.xlim([0, fs/2])
plt.show()



# %%











