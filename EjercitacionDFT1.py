import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

N=8
X=np.zeros(N, dtype=np.complex128)
n = np.arange(N)
x = 3*np.sin(n*np.pi/2) + 4

for k in range(N):
    for n in range(N):
        X[k] += x[n] * np.exp(-1j*k*2*np.pi*n/N)
        
        
print(X)

plt.figure(1)
markerline, stemlines, baseline = plt.stem(np.arange(N), np.abs(X)) #abs para graficar el modulo
plt.setp(markerline, color='orchid')       # puntitos 
plt.setp(stemlines, color='orchid')      # palitos 
plt.setp(baseline, color='lightseagreen')        # base
plt.title("Espectro (DFT)")
plt.xlabel("Índice k")
plt.ylabel("|X[k]|")
plt.grid(True)





