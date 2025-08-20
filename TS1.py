import numpy as np  
import matplotlib.pyplot as plt

def mi_funcion_sen( vmax, dc, ff, ph, nn, fs):
    #vmax:amplitud max de la senoidal [Volts]
    #dc:valor medio [Volts]
    #ff:frecuencia [Hz]
    #ph:fase en [rad]
    #nn:cantidad de muestras
    #fs:frecuencia de muestreo [Hz]

    Ts = 1/fs #período de muestre (osea cuanto tiempo hay entre muestra y muestra)
    tt = np.linspace(0, (nn-1)*Ts, nn) #tt seria el vector de tiempos donde se evalúa la señal
    #np.linspace(start, stop, num, endpoint=True, retstep=False, dtype=None, axis=0)
    xx=vmax*np.sin(2*np.pi*ff*tt+ph)+dc #np.pi es pi y np.sin es para hacer el seno

    return tt,xx 

N=1000 
fs=1000
tt, yy = mi_funcion_sen(1, 0, 2000, 0, N, fs)
plt.figure(1)
plt.plot(tt, yy, color='orchid') 
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()