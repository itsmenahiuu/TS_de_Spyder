import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    # vmax:amplitud max de la senoidal [Volts]
    # dc:valor medio [Volts]
    # ff:frecuencia [Hz]
    # ph:fase en [rad]
    # nn:cantidad de muestras
    # fs:frecuencia de muestreo [Hz]

    # período de muestre (osea cuanto tiempo hay entre muestra y muestra)
    Ts = 1/fs
    # tt seria el vector de tiempos donde se evalúa la señal
    tt = np.linspace(0, (nn-1)*Ts, nn)
    # np.linspace(start, stop, num, endpoint=True, retstep=False, dtype=None, axis=0)
    # np.pi es pi y np.sin es para hacer el seno
    xx = vmax*np.sin(2*np.pi*ff*tt+ph)+dc

    return tt, xx


N = 1000  # bajo las muestras para que se vea mejor la senoidal)
fs = 80000  # debo poner al menos el doble de la frecuencia que quiero esstudiar
# use 2kHz asi que al menos mi frec de muestreo deeria ser 4kHz pero puse 80k para darme mas margen
tt, xx = mi_funcion_sen(1, 0, 2000, 0, N, fs)
plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(tt, xx, color='orchid')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.xlim(0, 0.003)
plt.show()

#-----------SEÑAL AMPLIFICADA Y DESFASADA PI/2 (de amplitud 1 a 5)-----------#
tt1, xx1 = mi_funcion_sen(6, 0, 2000, np.pi/2, N, fs)
plt.subplot(1, 2, 2)
plt.plot(tt1, xx1, color='rebeccapurple')
plt.title('Señal Senoidal')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.xlim(0, 0.003)
plt.show()

plt.tight_layout()
plt.show()


#-----------------SEÑAL MODULADA-----------------#
tt, xx = mi_funcion_sen(1, 0, 2000, 0, N, fs)
ttm, xxm = mi_funcion_sen(1, 0, 1000, 0, N, fs)
modulada = xx*xxm
plt.figure(2)
plt.plot(tt, xx, color='skyblue')
plt.plot(ttm, xxm, color='lightseagreen')
plt.plot(ttm, modulada, color='pink', linewidth=2)
plt.title('Señal Modulada y sus partes')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.xlim(0, 0.0017)

plt.tight_layout()
plt.show()

#-----------------SEÑAL RECORTADA-----------------#
ttR, xxR = mi_funcion_sen(1, 0, 2000, 0, N, fs)
potencia = np.mean(xxR**2)
pot75 = potencia*0.75
amplitudRecortada = np.sqrt(pot75*2)
senoidalRecortada = np.clip(xxR, -amplitudRecortada, amplitudRecortada)
plt.figure(3)
plt.plot(ttR, xxR, color='rebeccapurple')
plt.plot(ttR, senoidalRecortada, color='skyblue')
plt.title('Señal Recortada en un 75% de su Potencia')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.xlim(0, 0.003)

#-----------------SEÑAL CUADRADA-----------------#
TsC = 1/fs
ttC = np.linspace(0, (N-1)*TsC, N)
señalCuadrada = sig.square(2 * np.pi * 4000 * ttC)

plt.figure(4)
plt.subplot(1, 2, 1)
plt.plot(ttm, señalCuadrada, color='lightseagreen')
plt.title('Señal Cuadrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.xlim(0, 0.0010)

plt.tight_layout()
plt.show()

#-----------------PULSO RECTANGULAR-----------------#
TsP = 0.01   # 10 ms
fsP = 80000   
#t = np.linspace(0, Ttotal, N, endpoint=False)
t = np.arange(-2*TsP, 2*TsP, 1/fsP)  # eje temporal de -20 ms a 20 ms
pulso = np.where((t >= -TsP/2) & (t <= TsP/2), 1, 0)#vale 1 entre -T/2 y T/2, 0 fuera

#--> True (1)
# --> False (0)

# Graficar
plt.subplot(1, 2, 2)
plt.plot(t*1000, pulso, drawstyle='steps-post',color='deeppink' ) 
#hice t*1000 para que este en ms
#drawstyle='steps-post' es para que se vea como un pulso y no un triangulo
plt.xlabel("Tiempo [ms]")
plt.ylabel("Amplitud")
plt.title("Pulso rectangular de 10 ms")
plt.grid(True)

plt.tight_layout()
plt.show()


#-----------------ORTOGONALIDAD-----------------#
def verificarOrtogonalidad(x1, x2):

    L = min(len(x1), len(x2)) # Cuando hace producto interno necesitas tener la misma cant de puntos
    #Por si no tenes la misma cantidad de puntos, pones eso y toma la cant de puntos de la señal qie menos puntos tenga
    productoInterno = np.dot(x1[:L], x2[:L])
    return productoInterno


producto = verificarOrtogonalidad(xx, xx1)
print("Verificacion de ortogonalidad entre mi senoidal de 2KHz y la misma amplificada y desfasada: \n", round(producto, 3))
if abs(producto) < 1e-10: #porque no da 0 justo por alguna razon
    print("Las señales son ortogonales\n")
else:
    print("Las señales no son ortogonales\n")


producto = verificarOrtogonalidad(xx, modulada)
print("Verificacion de ortogonalidad entre mi senoidal de 2KHz y una modulada por una sinusoidal de la mitad de la frecuencia: \n", round(producto, 3))
if abs(producto) < 1e-10: #porque no da 0 justo por alguna razon
    print("Las señales son ortogonales\n")
else:
    print("Las señales no son ortogonales\n")


producto = verificarOrtogonalidad(xx, senoidalRecortada)
print("Verificacion de ortogonalidad entre mi senoidal de 2KHz y la misma recortada al 75% de su amplitud: \n", round(producto, 3))
if abs(producto) < 1e-10: #porque no da 0 justo por alguna razon 
    print("Las señales son ortogonales\n")
else:
    print("Las señales no son ortogonales\n")


producto = verificarOrtogonalidad(xx, señalCuadrada)
print("Verificacion de ortogonalidad entre mi senoidal de 2KHz y una señal cuadrada de 4KHz: \n", round(producto, 3))
if abs(producto) < 1e-10: #porque no da 0 justo por alguna razon
    print("Las señales son ortogonales\n")
else:
    print("Las señales no son ortogonales\n")


producto = verificarOrtogonalidad(xx, pulso)
print("Verificacion de ortogonalidad entre mi senoidal de 2KHz y un pulso rectangular de 10ms: \n", round(producto, 3))
if abs(producto) < 1e-10: #porque no da 0 justo por alguna razon
    print("Las señales son ortogonales\n")
else:
    print("Las señales no son ortogonales\n")


#-----------------CORRELACION-----------------#

def correlacion(x1, x2, mode="full"):

    demora = sig.correlation_lags(len(x1), len(x2), mode=mode) #Calculo los desplazamientos correspondientes a cada valor que devuelve wl correlate
    # osea indico cuanto se desplazo la señal "x1" respecto a "x2".
    C = sig.correlate(x1, x2, mode=mode) #Devuelvo un vector de valores de correlacion para cada posible desplazamiento
    #La correlacion mide que tan parecidas son las señales cuando una se desplaza respecto a la otra
    
    return demora, C


demora, C = correlacion(xx, xx) 
plt.figure(5)
plt.subplot(3, 2, 1)
plt.plot(demora, C, color='mediumvioletred')
plt.title("Autocorrelacion")
plt.xlabel("Demora")
plt.ylabel("Correlacion")
plt.grid(True)

plt.tight_layout()
plt.show()

demora1, C1 = correlacion(xx, xx1) 
plt.subplot(3, 2, 2)
plt.plot(demora1, C1, color='deepskyblue')
plt.title("Correlacion con otra senoidal")
plt.xlabel("Demora")
plt.ylabel("Correlacion")
plt.grid(True)

plt.tight_layout()
plt.show()

demora2, C2 = correlacion(xx, modulada) 
plt.subplot(3, 2, 3)
plt.plot(demora2, C2, color='indigo')
plt.title("Correlacion con una señal modulada")
plt.xlabel("Demora")
plt.ylabel("Correlacion")
plt.grid(True)

plt.tight_layout()
plt.show()

demora3, C3 = correlacion(xx, senoidalRecortada) 
plt.subplot(3, 2, 4)
plt.plot(demora3, C3, color='teal')
plt.title("Correlacion con una señal recortada")
plt.xlabel("Demora")
plt.ylabel("Correlacion")
plt.grid(True)

plt.tight_layout()
plt.show()

demora4, C4 = correlacion(xx, señalCuadrada) 
#plt.figure(6)
plt.subplot(3, 2, 5)
plt.plot(demora4, C4, color='gold')
plt.title("Correlacion con una señal cuadrada")
plt.xlabel("Demora")
plt.ylabel("Correlacion")
plt.grid(True)

plt.tight_layout()
plt.show()

demora5, C5 = correlacion(xx, pulso) 
#plt.figure(7)
plt.subplot(3, 2, 6)
plt.plot(demora5, C5, color='plum')
plt.title("Correlacion con un pulso rectangular")
plt.xlabel("Demora")
plt.ylabel("Correlacion")
plt.grid(True)

plt.tight_layout()
plt.show()





