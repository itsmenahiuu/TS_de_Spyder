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

señalConRuidoFFT = fft(señalConRuidoMatriz, axis=0)/N #(osea sin ventanita)
espectroRect = 10*np.log10(2*np.abs(señalConRuidoFFT)**2)

a2 = 10*np.log10((np.abs(señalConRuidoFFT[N//4,:])**2)*2)


#HAMMING
hamming = win.hamming(N).reshape(N,1)
señalRuidosaHamming = señalConRuidoMatriz * hamming
señalRuidosaHammingFFT = fft(señalRuidosaHamming, axis=0)/N
espectroHamming = 10*np.log10(2*np.abs(señalRuidosaHammingFFT)**2)

a3 = 10*np.log10((np.abs(señalRuidosaHammingFFT[N//4,:])**2)*2)

#FLATTOP
flattop = win.flattop(N).reshape(N,1)
señalRuidosaFlattop = señalConRuidoMatriz * flattop
señalRuidosaFlattopFFT = fft(señalRuidosaFlattop, axis=0)/N
espectroFlattop = 10*np.log10(2*np.abs(señalRuidosaFlattopFFT)**2)

a4 = 10*np.log10((np.abs(señalRuidosaFlattopFFT[N//4,:])**2)*2)

#BLACKMANHARRIS
blackmanH = win.blackmanharris(N).reshape(N,1)
señalRuidosaBlackH = señalConRuidoMatriz * blackmanH
señalRuidosaBlackHFFT = fft(señalRuidosaBlackH, axis=0)/N
espectroBlackmanH = 10*np.log10(2*np.abs(señalRuidosaBlackHFFT)**2)

a5 = 10*np.log10((np.abs(señalRuidosaBlackHFFT[N//4,:])**2)*2)


# señalConRuidoFFT = fft(señalConRuidoMatriz, axis=0)/N #(osea sin ventanita)
# a1 = 10*np.log10((np.abs(señalConRuidoFFT[N//4,:])**2)*2)


#para graficar me armo el eje de frecuencias
freqs = np.linspace(0, fs, N)

# %%


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

# %%


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

# %%


plt.figure(3)
plt.plot(freqs, espectroFlattop)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana Flattop")
plt.xlim([0, fs/2])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%


plt.figure(4)
plt.plot(freqs, espectroBlackmanH)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud [dB]")
plt.title("Ventana Blackman Harris")
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

# %%

#HISTOGRAMAS
plt.figure(6)
bins = 10
plt.hist(a2,label='Rectangular', bins = bins, color = 'cornflowerblue')
plt.hist(a3,label='Hamming', alpha = 0.7, bins = bins, color = 'hotpink')
plt.hist(a5,label='Blackman Harris', alpha = 0.5, bins = bins, color = 'indigo')
plt.hist(a4,label='Flattop', alpha = 0.6, bins = bins, color = 'limegreen')
plt.legend()
plt.show()

