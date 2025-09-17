
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


fs = 8000    
N = 1000     

# Sistema A: y[n] = x[n] + 3*x[n-10]
bA = np.zeros(11)   
bA[0] = 1
bA[10] = 3
aA = np.array([1])  

# Sistema B: y[n] = x[n] + 3*y[n-10]
bB = np.array([1])  
aB = np.zeros(11)
aB[0] = 1
aB[10] = -3   

# Sistema C: (ejemplo simple, un promedio móvil)
bC = np.array([0.03, 0.05, 0.03]) #para lo que tiene x
aC = np.array([1, -1.5, 0.5]) #para lo que tiene y

#freqz calcula la respuesta H(e^(jω)) de cada sistema
wA, HA = sig.freqz(bA, aA, worN=4096)
wB, HB = sig.freqz(bB, aB, worN=4096)
wC, HC = sig.freqz(bC, aC, worN=4096)

#convertir ω a Hz
freqsA = wA * fs / (2*np.pi)
freqsB = wB * fs / (2*np.pi)
freqsC = wC * fs / (2*np.pi)

plt.figure(1)

plt.plot(freqsA, 20*np.log10(np.abs(HA)), label="Sistema A", color="orchid")
plt.plot(freqsB, 20*np.log10(np.abs(HB)), label="Sistema B", color="cornflowerblue")
plt.plot(freqsC, 20*np.log10(np.abs(HC)), label="Sistema C", color="yellowgreen")

plt.title("Respuesta en Frecuencia (Magnitud)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid()
plt.legend()
plt.show()

