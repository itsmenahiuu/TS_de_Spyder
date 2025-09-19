import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal as sig
win = sig.windows


N=1000
fs=N

def SNRvarianza(señal, SNRdb):
    # Potencia de la señal
    potSeñal = np.mean(señal**2)

    # Varianza del ruido
    Rvar = potSeñal / (10**(SNRdb / 10))

    return Rvar

def sen( vmax, dc, ff, ph, nn, fs): 

    Ts = 1/fs 
    tt = np.linspace(0, (nn-1)*Ts, nn) 

    xx=vmax*np.sin(2*np.pi*ff*tt+ph)+dc 

    return tt,xx

t = np.arange(0, N)*(1/fs)

t, señal = sen(np.sqrt(2), 0, fs/4, 0, N, fs)

Rvar = SNRvarianza(señal, 10)
ruido = np.random.normal(0,np.sqrt(Rvar),N)

salida = señal + ruido
# varx = np.var(señal)
# varSalida = np.var(salida)
# print("Varianza: ", Rvar)
# print("Varianza: ", varx)
# print("Varianza: ", varSalida)

SEÑALfft = fft(señal)
SALIDAfft = fft(salida)


plt.figure()
plt.plot(10*np.log10(2*np.abs(SALIDAfft*(1/N))**2), label = "Señal + Ruido", color = 'mediumvioletred')
#plt.plot(20*np.log10(np.abs(SEÑALfft)), color = 'pink', label = "Señal")
plt.xlabel('Muestras [rad]')
plt.ylabel('Amplitud [dB]')
plt.title('Señal con SNR')
plt.legend()
plt.grid(True)
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal as sig
win = sig.windows


N=1000
R=200
fs=N
frecRandom= np.random.uniform(low=-2, high=2, size=R) #ojo que es un vector
frecRandom = frecRandom.reshape(1, R)

def SNRvarianza(señal, SNRdb):
    # Potencia de la señal
    potSeñal = np.mean(señal**2)

    # Varianza del ruido
    Rvar = potSeñal / (10**(SNRdb / 10))

    return Rvar

def sen( vmax, dc, ff, ph, nn, fs): 

    Ts = 1/fs 
    tt = np.linspace(0, (nn-1)*Ts, nn) 

    xx=vmax*np.sin(2*np.pi*ff*tt+ph)+dc 

    return tt,xx

t = np.arange(0, N)*(1/fs)


#quiero una matriz
t, señal = sen(np.sqrt(2), 0, fs/4, 0, N, fs)
señales = np.tile(señal, reps=(N, R))   

Rvar = SNRvarianza(señal, 10)
ruido = np.random.normal(0,np.sqrt(Rvar),size=(N, R))

salida = señal + ruido
# varx = np.var(señal)
# varSalida = np.var(salida)
# print("Varianza: ", Rvar)
# print("Varianza: ", varx)
# print("Varianza: ", varSalida)

SEÑALfft = fft(señal)
SALIDAfft = fft(salida, axis=0) #axis=0 es filas


plt.figure()
plt.plot(10*np.log10(2*np.abs(SALIDAfft*(1/N))**2), label = "Señal + Ruido", color = 'mediumvioletred')
#plt.plot(20*np.log10(np.abs(SEÑALfft)), color = 'pink', label = "Señal")
plt.xlabel('Muestras [rad]')
plt.ylabel('Amplitud [dB]')
plt.title('Señal con SNR')
plt.legend()
plt.grid(True)
plt.show()





# %%
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal as sig
win = sig.windows

# Parámetros
N = 1000      #muestras
R = 200       #realizaciones
fs = N        #frecuencia de muestreo 
f0 = fs / 4   #frecuencia de la senoidal
resolucionEspectral=fs/N
frecRandom= np.random.uniform(low=-2, high=2, size=R)*resolucionEspectral #ojo que es un vector
frecRandom = frecRandom.reshape(1, R)

#tiempo en vector y matriz
Ts = 1/fs
t = np.linspace(0, (N-1)*Ts, N)            #vector de tiempo 
t_mat = np.tile(t.reshape(N, 1), (1, R))   #matriz 1000x200 (columnas = realizaciones)

#señal senoidal en matriz
a0 = np.sqrt(2)
señalMatriz = a0 * np.sin(2*np.pi*(f0+frecRandom)*t_mat)   # matriz 1000x200

def SNRvarianza(señal, SNRdb):
    potSeñal = np.mean(señal**2)
    return potSeñal / (10**(SNRdb / 10))

Rvar = SNRvarianza(señalMatriz, 10)

#rudio pero matriz
ruidoMatriz = np.random.normal(0, np.sqrt(Rvar), (N, R))

#señal+ruido
señalConRuidoMatriz = señalMatriz + ruidoMatriz


#RECTANGULAR
rect = np.ones((N,1))
señalRuidosaRect = señalConRuidoMatriz * rect  
señalRuidosaRectFFT = fft(señalRuidosaRect, axis=0)/N
espectroRect = 10*np.log10(2*np.abs(señalRuidosaRectFFT)**2)

a2 = 10*np.log10((np.abs(señalRuidosaRectFFT[N//4,:])**2)*2)


#HAMMING
hamming = win.hamming(N).reshape(N,1)
señalRuidosaHamming = señalConRuidoMatriz * hamming
señalRuidosaHammingFFT = fft(señalRuidosaHamming, axis=0)/N
espectroHamming = 10*np.log10(2*np.abs(señalRuidosaHammingFFT)**2)

a3 = 10*np.log10((np.abs(señalRuidosaHammingFFT[N//4,:])**2)*2)

#HANN
hann = win.hann(N).reshape(N,1)
señalRuidosaHann = señalConRuidoMatriz * hann
señalRuidosaHannFFT = fft(señalRuidosaHann, axis=0)/N
espectroHann = 10*np.log10(2*np.abs(señalRuidosaHannFFT)**2)

a4 = 10*np.log10((np.abs(señalRuidosaHannFFT[N//4,:])**2)*2)

#BLACKMAN
blackman = win.blackman(N).reshape(N,1)
señalRuidosaBlack = señalConRuidoMatriz * blackman
señalRuidosaBlackFFT = fft(señalRuidosaBlack, axis=0)/N
espectroBlackman = 10*np.log10(2*np.abs(señalRuidosaBlackFFT)**2)

a5 = 10*np.log10((np.abs(señalRuidosaBlackFFT[N//4,:])**2)*2)


# señalConRuidoFFT = fft(señalConRuidoMatriz, axis=0)/N #(osea sin ventanita)
# a1 = 10*np.log10((np.abs(señalConRuidoFFT[N//4,:])**2)*2)


#para graficar me armo el eje de frecuencias
freqs = np.linspace(0, fs, N)

plt.figure(1)
plt.plot(freqs, espectroRect)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana Rectangular")
plt.xlim([0, fs/2])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(2)
plt.plot(freqs, espectroHamming)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana Hamming")
plt.xlim([0, fs/2])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(3)
plt.plot(freqs, espectroHann)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana Hann")
plt.xlim([0, fs/2])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



plt.figure(4)
plt.plot(freqs, espectroBlackman)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana Blackman")
plt.xlim([0, fs/2])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# plt.figure(5)
# plt.plot(señalConRuidoMatriz[0, :], label="Fila 0")
# plt.plot(señalConRuidoMatriz[N//4, :], label=f"Fila {N//4}")
# plt.plot(señalConRuidoMatriz[N//2, :], label=f"Fila {N//2}")
# plt.plot(señalConRuidoMatriz[N-1, :], label=f"Fila {N-1}")

# plt.title("Filitas de la matriz")
# plt.xlabel("Realizacion (columna)")
# plt.ylabel("Amplitud")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()


plt.figure(6)
bins = 10
plt.hist(a2,label='Rectangular', bins = bins, color = 'cornflowerblue')
plt.hist(a3,label='Hamming', alpha = 0.7, bins = bins, color = 'hotpink')
plt.hist(a4,label='Hann', alpha = 0.5, bins = bins, color = 'limegreen')
plt.hist(a5,label='Blackman', alpha = 0.3, bins = bins, color = 'darkorchid')
plt.legend()
plt.show()





