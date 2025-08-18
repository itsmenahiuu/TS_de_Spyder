#llamo a mi libreria numpy (como en c)
import numpy as np 
import matplotlib.pyplot as plt

def mi_funcion_sen( vmax, dc, ff, ph, nn, fs): 
#defino (def) mi funcion para pasarle todos los datos

#vmax:amplitud max de la senoidal [Volts]
#dc:valor medio [Volts]
#ff:frecuencia [Hz]
#ph:fase en [rad]
#nn:cantidad de muestras
#fs:frecuencia de muestreo [Hz]

    Ts = 1/fs #período de muestre (osea cuanto tiempo hay entre muestra y muestra)
    tt = np.linspace(0, (nn-1)*Ts, nn) #tt seria el vector de tiempos donde se evalúa la señal
#np.linspace(start, stop, num, endpoint=True, retstep=False, dtype=None, axis=0)

# - start: valor inicial
# - stop: valor final
# - num: cantidad de puntos (default = 50)
# - endpoint: True -> incluye el valor final; False -> no lo incluye
# - retstep: True -> devuelve tambien el paso entre puntos
# - dtype: tipo de dato (float, int, etc.)
# - axis: eje en el que se coloca el resultado en arrays multidimensionales
 
    xx=vmax*np.sin(2*np.pi*ff*tt+ph)+dc #np.pi es pi y np.sin es para hacer el seno

    return tt,xx #cierro mi funcion "def", es como las funciones de C pero no tiene llaves, es asi nomas
#se tabula todo lo que esta dentro de mi funcion 

N=1000
fs=1000
tt, yy = mi_funcion_sen(1, 0, 1, 0, N, fs)
plt.figure(1)
plt.plot(tt, yy, color='orchid') 
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.show()


#Bonus

#cambio mi ff por las del bonus
tt1, yy1 = mi_funcion_sen(1, 0, 500, 0, 1000, 1000)
tt2, yy2 = mi_funcion_sen(1, 0, 999, 0, 1000, 1000)
tt3, yy3 = mi_funcion_sen(1, 0, 1001, 0, 1000, 1000)
tt4, yy4 = mi_funcion_sen(1, 0, 2001, 0, 1000, 1000)

plt.figure(2)
plt.subplot(2,2,1) #como en matlab
plt.plot(tt1, yy1, color='pink') 
plt.title('ff = 500 Hz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')

plt.subplot(2,2,2)
plt.plot(tt2, yy2, color='deeppink') 
plt.title('ff = 999 Hz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')

plt.subplot(2,2,3)
plt.plot(tt3, yy3, color='mediumorchid') 
plt.title('ff = 1001 Hz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')

plt.subplot(2,2,4)
plt.plot(tt4, yy4, color='rebeccapurple') 
plt.title('ff = 2001 Hz')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')

plt.tight_layout()  #para que no se solapen mis graficos (solucion que me dio gtp a un error que tenia)
plt.show()




