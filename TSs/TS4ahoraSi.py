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
a0 = 2
señalMatriz = a0 * np.sin(2*np.pi*(f0+frecRandom)*t_mat)   # matriz 1000x200

def SNRvarianza(señal, SNRdb):
    potSeñal = np.mean(señal**2)
    return potSeñal / (10**(SNRdb / 10))

Rvar = SNRvarianza(señalMatriz, 10)

#rudio pero matriz
ruidoMatriz = np.random.normal(0, np.sqrt(Rvar), (N, R))

#señal+ruido
señalConRuidoMatriz = señalMatriz + ruidoMatriz

# %%

#Ventanas

#---------------RECTANGULAR---------------#

señalConRuidoFFT = fft(señalConRuidoMatriz, axis=0)/N #(osea sin ventanita)
espectroRect = 10*np.log10(2*np.abs(señalConRuidoFFT)**2)

#---------------HAMMING---------------#
hamming = win.hamming(N).reshape(N,1)
señalRuidosaHamming = señalConRuidoMatriz * hamming
señalRuidosaHammingFFT = fft(señalRuidosaHamming, axis=0)/N
espectroHamming = 10*np.log10(2*np.abs(señalRuidosaHammingFFT)**2)


#---------------FLATTOP---------------#
flattop = win.flattop(N).reshape(N,1)
señalRuidosaFlattop = señalConRuidoMatriz * flattop
señalRuidosaFlattopFFT = fft(señalRuidosaFlattop, axis=0)/N
espectroFlattop = 10*np.log10(2*np.abs(señalRuidosaFlattopFFT)**2)


#---------------BLACKMANHARRIS---------------#
blackmanH = win.blackmanharris(N).reshape(N,1)
señalRuidosaBlackH = señalConRuidoMatriz * blackmanH
señalRuidosaBlackHFFT = fft(señalRuidosaBlackH, axis=0)/N
espectroBlackmanH = 10*np.log10(2*np.abs(señalRuidosaBlackHFFT)**2)


# %%

#---------------ESTIMADOR DE AMPLITUD---------------#


#RECTANGULAR
a2 = 10*np.log10((np.abs(señalConRuidoFFT[N//4,:])**2)*2)  #estimador
a2_lin = np.sqrt(10**(a2/10))  # paso de dB a valor a lineal

a2prom = np.mean(a2_lin) #promedio de las amplitudes estimadas
sesgo_a2 = a2prom - a0  
var_a2 = np.var(a2_lin) 

#HAMMING
a3 = 10*np.log10((np.abs(señalRuidosaHammingFFT[N//4,:])**2)*2)  #estimador 
a3_lin = np.sqrt(10**(a3/10))

a3prom = np.mean(a3_lin)
sesgo_a3 = a3prom - a0
var_a3 = np.var(a3_lin)

#FLATTOP
a4 = 10*np.log10((np.abs(señalRuidosaFlattopFFT[N//4,:])**2)*2)  #estimador 
a4_lin = np.sqrt(10**(a4/10))

a4prom = np.mean(a4_lin)
sesgo_a4 = a4prom - a0
var_a4 = np.var(a4_lin)

#BLACKMANHARRIS
a5 = 10*np.log10((np.abs(señalRuidosaBlackHFFT[N//4,:])**2)*2)  #estimador
a5_lin = np.sqrt(10**(a5/10))

a5prom = np.mean(a5_lin)
sesgo_a5 = a5prom - a0
var_a5 = np.var(a5_lin)


print("\nESTIMADOR DE AMPLITUD SNR:10dB")
print("--------------------------------")
print("VENTANA         SESGO_AMP      VAR_AMP")
print(f"Rectangular     {sesgo_a2:.4f}      {var_a2:.4f}")
print(f"Hamming         {sesgo_a3:.4f}      {var_a3:.4f}")
print(f"Flattop         {sesgo_a4:.4f}      {var_a4:.4f}")
print(f"Blackman-Harris {sesgo_a5:.4f}      {var_a5:.4f}")


#HISTOGRAMAS
bins = 30

# ----------HISTOGRAMA DE AMPLITUD---------- #
plt.figure(1)
plt.hist(a2_lin,label='Rectangular', bins = bins, color = 'cornflowerblue')
plt.hist(a3_lin,label='Hamming', alpha = 0.7, bins = bins, color = 'hotpink')
plt.hist(a5_lin,label='Blackman Harris', alpha = 0.5, bins = bins, color = 'indigo')
plt.hist(a4_lin,label='Flattop', alpha = 0.6, bins = bins, color = 'limegreen')
plt.axvline(a0, color="k", linestyle="--", label="Amplitud real")
plt.xlabel("Frecuencia estimada (Hz)")
plt.ylabel("Número de realizaciones")
plt.legend()
plt.grid(True)
plt.show()


# %%

#---------------ESTIMADOR DE FRECUENCIA---------------#

# Vector de frecuencias (Hz)
frecuencias = np.fft.fftfreq(N, d=1/fs)
frecuenciasPositivas = frecuencias[:N//2]   # solo la mitad positiva asi no tengo los dos picpos

# ---------------- Rectangular ---------------- #
indiceMaximoRectangular = np.argmax(np.abs(señalConRuidoFFT[:N//2, :]), axis=0)
omega1 = frecuenciasPositivas[indiceMaximoRectangular]   # estimador de frecuencia
sesgoOmega1 = np.mean(omega1) - f0
varianzaOmega1 = np.var(omega1)

# ---------------- Hamming ---------------- #
indiceMaximoHamming = np.argmax(np.abs(señalRuidosaHammingFFT[:N//2, :]), axis=0)
omega2 = frecuenciasPositivas[indiceMaximoHamming]
sesgoOmega2 = np.mean(omega2) - f0
varianzaOmega2 = np.var(omega2)

# ---------------- Flattop ---------------- #
indiceMaximoFlattop = np.argmax(np.abs(señalRuidosaFlattopFFT[:N//2, :]), axis=0)
omega3 = frecuenciasPositivas[indiceMaximoFlattop]
sesgoOmega3 = np.mean(omega3) - f0
varianzaOmega3 = np.var(omega3)

# ---------------- Blackman-Harris ---------------- #
indiceMaximoBlackmanHarris = np.argmax(np.abs(señalRuidosaBlackHFFT[:N//2, :]), axis=0)
omega4 = frecuenciasPositivas[indiceMaximoBlackmanHarris]
sesgoOmega4 = np.mean(omega4) - f0
varianzaOmega4 = np.var(omega4)

#---------------------------TABLA DE RESULTADOS---------------------------#
print("\nESTIMADOR DE FRECUENCIA SNR:10dB")
print("VENTANA          SESGO_FREQ        VAR_FREQ")
print(f"Rectangular      {sesgoOmega1:.4f}        {varianzaOmega1:.4f}")
print(f"Hamming          {sesgoOmega2:.4f}        {varianzaOmega2:.4f}")
print(f"Flattop          {sesgoOmega3:.4f}        {varianzaOmega3:.4f}")
print(f"Blackman-Harris  {sesgoOmega4:.4f}        {varianzaOmega4:.4f}")

#---------------------------HISTOGRAMA DE FREC---------------------------#
plt.figure(2)
plt.hist(omega1, bins=50, alpha=0.5, label="Rectangular", color = 'cornflowerblue')
plt.hist(omega2, bins=50, alpha=0.5, label="Hamming", color = 'hotpink')
plt.hist(omega3, bins=50, alpha=0.5, label="Flattop", color = 'indigo')
plt.hist(omega4, bins=50, alpha=0.5, label="Blackman-Harris", color = 'limegreen')
plt.axvline(f0, color="k", linestyle="--", label="Frecuencia real")
plt.xlabel("Frecuencia estimada (Hz)")
plt.ylabel("Número de realizaciones")
plt.title("Histogramas de estimadores de frecuencia")
plt.legend()
plt.grid(True)
plt.show()

# %%

#BONUS

N_zp = 4 * N  
frecuencias_zp = np.fft.fftfreq(N_zp, d=1/fs)
frecuenciasPositivas_zp = frecuencias_zp[:N_zp//2]

# ---------------- Rectangular ---------------- #
señalConRuidoFFT_zp = fft(señalConRuidoMatriz, n=N_zp, axis=0)/N
indiceMaximoRectangular_zp = np.argmax(np.abs(señalConRuidoFFT_zp[:N_zp//2, :]), axis=0)
omega1_zp = frecuenciasPositivas_zp[indiceMaximoRectangular_zp]
sesgoOmega1_zp = np.mean(omega1_zp) - f0
varianzaOmega1_zp = np.var(omega1_zp)

# ---------------- Hamming ---------------- #
señalRuidosaHammingFFT_zp = fft(señalRuidosaHamming, n=N_zp, axis=0)/N
indiceMaximoHamming_zp = np.argmax(np.abs(señalRuidosaHammingFFT_zp[:N_zp//2, :]), axis=0)
omega2_zp = frecuenciasPositivas_zp[indiceMaximoHamming_zp]
sesgoOmega2_zp = np.mean(omega2_zp) - f0
varianzaOmega2_zp = np.var(omega2_zp)

# ---------------- Flattop ---------------- #
señalRuidosaFlattopFFT_zp = fft(señalRuidosaFlattop, n=N_zp, axis=0)/N
indiceMaximoFlattop_zp = np.argmax(np.abs(señalRuidosaFlattopFFT_zp[:N_zp//2, :]), axis=0)
omega3_zp = frecuenciasPositivas_zp[indiceMaximoFlattop_zp]
sesgoOmega3_zp = np.mean(omega3_zp) - f0
varianzaOmega3_zp = np.var(omega3_zp)

# ---------------- Blackman-Harris ---------------- #
señalRuidosaBlackHFFT_zp = fft(señalRuidosaBlackH, n=N_zp, axis=0)/N
indiceMaximoBlackmanHarris_zp = np.argmax(np.abs(señalRuidosaBlackHFFT_zp[:N_zp//2, :]), axis=0)
omega4_zp = frecuenciasPositivas_zp[indiceMaximoBlackmanHarris_zp]
sesgoOmega4_zp = np.mean(omega4_zp) - f0
varianzaOmega4_zp = np.var(omega4_zp)

# ------------------ Tabla de resultados ------------------ #
print("\nESTIMADOR DE FRECUENCIA con ZERO-PADDING (SNR=10dB)")
print("VENTANA          SESGO_FREQ        VAR_FREQ")
print(f"Rectangular      {sesgoOmega1_zp:.4f}        {varianzaOmega1_zp:.4f}")
print(f"Hamming          {sesgoOmega2_zp:.4f}        {varianzaOmega2_zp:.4f}")
print(f"Flattop          {sesgoOmega3_zp:.4f}        {varianzaOmega3_zp:.4f}")
print(f"Blackman-Harris  {sesgoOmega4_zp:.4f}        {varianzaOmega4_zp:.4f}")

# ------------------ Histogramas ------------------ #
plt.figure()
plt.hist(omega1_zp, bins=50, alpha=0.5, label="Rectangular")
plt.hist(omega2_zp, bins=50, alpha=0.5, label="Hamming")
plt.hist(omega3_zp, bins=50, alpha=0.5, label="Flattop")
plt.hist(omega4_zp, bins=50, alpha=0.5, label="Blackman-Harris")
plt.axvline(f0, color="k", linestyle="--", label="Frecuencia real")
plt.xlabel("Frecuencia estimada (Hz)")
plt.ylabel("Número de realizaciones")
plt.title("Histogramas de estimadores de frecuencia con Zero-Padding")
plt.legend()
plt.grid(True)
plt.show()










