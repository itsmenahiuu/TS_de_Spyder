import numpy as np
import matplotlib.pyplot as plt


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
freqs = np.arange(0, N//2) * df

#-----------------N/4-----------------#
ff=(N/4)*df
_, yy = mi_funcion_sen(1, 0, ff, 0, N, fs)
#el _, me ignora tt, porque no lo necesito
FFT=np.fft.fft(yy)

plt.figure(1)
plt.stem(freqs, np.abs(FFT[:N//2]), linefmt="orchid", markerfmt="o", basefmt="orchid")

#-----------------N/4 + 1-----------------#
ff1=(N/4 + 1)*df
_, yy1 = mi_funcion_sen(1, 0, ff1, 0, N, fs)
FFT1=np.fft.fft(yy1)
plt.stem(freqs, np.abs(FFT1[:N//2]), linefmt="lightseagreen", markerfmt="o", basefmt="lightseagreen")      


#-----------------del medio-----------------#
ff2=(ff+ff1)/2
_, yy2 = mi_funcion_sen(1, 0, ff2, 0, N, fs)
FFT2=np.fft.fft(yy2)
plt.stem(freqs, np.abs(FFT2[:N//2]), linefmt="deepskyblue", markerfmt="o", basefmt="deepskyblue")

plt.title("FFT de senoidales")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X[k]|")
plt.grid(True)

#Si la frecuencia de la señal es múltiplo exacto de df, la FFT te da un solo palito
#sino, no




