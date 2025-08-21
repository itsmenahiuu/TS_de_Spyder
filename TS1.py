import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


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
# use 2kHz asi que al menos mi frec de muestreo deeria ser 4kHz pero puse 20k para darme mas margen
# tt, yy = mi_funcion_sen(1, 0, 2000, 0, N, fs)
# plt.figure(1)
# plt.subplot(1, 2, 1)
# plt.plot(tt, yy, color='orchid')
# plt.title('Señal Senoidal')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud [Volts]')
# plt.grid(True)
# plt.xlim(0, 0.005)
# plt.show()

# # SEÑAL AMPLIFICADA Y DESFASADA PI/2 (de amplitud 1 a 5)
# tt1, yy1 = mi_funcion_sen(6, 0, 2000, np.pi/2, N, fs)
# plt.subplot(1, 2, 2)
# plt.plot(tt1, yy1, color='rebeccapurple')
# plt.title('Señal Senoidal')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud [Volts]')
# plt.grid(True)
# plt.xlim(0, 0.005)
# plt.show()

# plt.tight_layout()
# plt.show()


# # SEÑAL MODULADA
Tsm = 1/fs
# # tt seria el vector de tiempos donde se evalúa la señal
ttm = np.linspace(0, (N-1)*Tsm, N)
# xx3 = np.sin(2*np.pi*2000*ttm)
# xx4 = np.sin(2*np.pi*1000*ttm)
# modulada = xx3*xx4
# plt.figure(2)
# plt.plot(ttm, xx3, color='skyblue')
# plt.plot(ttm, xx4, color='lightseagreen')
# plt.plot(ttm, modulada, color='pink')
# plt.title('Señal Modulada y sus partes')
# plt.xlabel('Tiempo [s]')
# plt.ylabel('Amplitud [Volts]')
# plt.grid(True)
# plt.xlim(0, 0.0017)

# plt.tight_layout()
# plt.show()

# #SEÑAL RECORTADA
# Ts5 = 1/fs
# # tt seria el vector de tiempos donde se evalúa la señal
# tt5 = np.linspace(0, (N-1)*Ts5, N)
# xx5 = np.sin(2*np.pi*2000*tt5) #amplitud 1
# potencia = np.mean(xx5**2)
# pot75 = potencia*0.75
# amplitudRecortada = np.sqrt(pot75*2)
# senoidalRecortada = np.clip(xx5, -amplitudRecortada, amplitudRecortada)
# plt.figure(3)
# plt.plot(ttm, xx3, color='orchid')
# plt.plot(ttm, senoidalRecortada, color='skyblue')

#SEÑAL CUADRADA
#cc = np.linspace(0, 1, fs, endpoint=False)  # vector de tiempo (1 s)
señalCuadrada = signal.square(2 * np.pi * 4000 * ttm)

plt.figure(4)
plt.plot(ttm, señalCuadrada, color='lightseagreen')
plt.title('Señal Cuadrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [Volts]')
plt.grid(True)
plt.xlim(0, 0.0010)