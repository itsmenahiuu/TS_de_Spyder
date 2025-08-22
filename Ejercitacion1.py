import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# N=32

# X=np.zeros(N, dtype=np.complex128)
# fs=N
# Ts = 1/fs
# ff=N/fs
# tt = np.linspace(0, (N-1)*Ts, N)
# x = np.sin(2*np.pi*ff*tt)

# # x = np.zeros(N)  #delta
# # x[0] = 1

# for k in range(N):
#     for n in range(N):
#         X[k] += x[n] * np.exp(-1j*k*2*np.pi*n/N)
        
        
# print(X)

# plt.figure(1)
# markerline, stemlines, baseline = plt.stem(np.arange(N), np.abs(X)) 
# plt.setp(markerline, color='orchid')       # puntitos 
# plt.setp(stemlines, color='orchid')      # palitos 
# plt.setp(baseline, color='lightseagreen')        # base
# plt.title("Espectro (DFT)")
# plt.xlabel("Índice k")
# plt.ylabel("|X[k]|")
# plt.grid(True)

# plt.figure(2)
# plt.stem(np.arange(N), np.angle(X))
# plt.title("Fase de la DFT")
# plt.xlabel("k")
# plt.ylabel("∠X[k] [rad]")
# plt.grid(True)
# plt.show()

#probar con un coseno, la proyeccion va a ser en los reales en vez de los imag
#si subo la frecuencia, se desplazan los palitos
#si me paso a N/2 (nyquiste) se me superponen las replicas



#-----------------AUTO CORRELACION-----------------#

M = 8
x = np.zeros(M)
y = np.zeros(M)
x[:2]=1
y[5]=1
rxx=sig.correlate(x,y)
plt.figure(3)
plt.plot(rxx, '0:', label= 'x')
plt.legend()










