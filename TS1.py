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

N=200 #bajo las muestras para que se vea mejor la senoidal)
fs=20000 #debo poner al menos el doble de la frecuencia que quiero esstudiar 
#use 2kHz asi que al menos mi frec de muestreo deeria ser 4kHz pero puse 20k para darme mas margen
tt, yy = mi_funcion_sen(1, 0, 2000, 0, N, fs)
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(tt, yy, color='orchid') 
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

#SEÑAL AMPLIFICADA Y DESFASADA PI/2 (de amplitud 1 a 5)
tt1, yy1 = mi_funcion_sen(6, 0, 2000, np.pi/2, N, fs)
plt.subplot(1,2,2)
plt.plot(tt1, yy1, color='rebeccapurple') 
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()

plt.tight_layout()  
plt.show()


#SEÑAL MODULADA
fsm=20000
Tsm = 1/fsm #período de muestre (osea cuanto tiempo hay entre muestra y muestra)
ttm = np.linspace(0, (N-1)*Tsm, N)
xx3=1*np.sin(2*np.pi*2000*ttm)
xxm=xx3*np.sin(2*np.pi*1000*ttm+np.pi/2)
plt.figure(2)
plt.plot(ttm, xxm, color='orchid') 
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)

plt.tight_layout()  
plt.show()
