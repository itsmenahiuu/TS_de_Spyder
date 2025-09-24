import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
plt.close('all')

#---------------Funciones auxiliares---------------#
def sen(vmax, dc, ff, ph, N, fs):
    """Genera una senoidal de amplitud vmax, offset dc, freq ff (Hz), fase ph (rad)."""
    Ts = 1/fs
    t = np.linspace(0, (N-1)*Ts, N)
    x = vmax*np.sin(2*np.pi*ff*t + ph) + dc
    return t, x

def potencia(x): 
    return np.mean(np.abs(x)**2)

def energia(x):  
    return np.sum(np.abs(x)**2)

#---------------Sistema---------------#
b = np.array([0.03, 0.05, 0.03]) #para lo que tiene x
a = np.array([1, -1.5, 0.5]) #para lo que tiene y

N = 1000
fs = 80000
Ts = 1/fs
Tsim = N/fs

print("Frecuencia de muestreo fs =", fs, "Hz")
print("Cantidad de muestras N =", N)
print("Tiempo de simulación T =", Tsim, "s")

#---------------mis señales del TS1---------------#
t, x  = sen(1, 0, 2000, 0, N, fs)
t1, x1 = sen(2, 0, 2000, np.pi/2, N, fs)
tm, xm = sen(1, 0, 1000, 0, N, fs)
modulada = x*xm
P = potencia(x)
amplitudRecortada = np.sqrt(2*0.75*P)
recortada = np.clip(x, -amplitudRecortada, amplitudRecortada)
tq = np.linspace(0, (N-1)/fs, N)
cuadrada = sig.square(2*np.pi*4000*tq)
TsP = 0.01   # 10 ms
fsP = 80000   
tp = np.linspace(0, Tsim, N, endpoint=False)
pulso = np.where((t >= 0) & (t <= TsP), 1, 0)

#---------------Entradas---------------#
entradas = [ #estpy literalmente armando un struct de c en python
    ("Senoidal 2 kHz", t, x),
    ("Senoidal amplificada y desfasada", t1, x1),
    ("Modulada", t, modulada),
    ("Recortada", t, recortada),
    ("Cuadrada 4 kHz", tq, cuadrada),
    ("Pulso rectangular 10 ms", tp, pulso),
]

#---------------Paso mis señales por la ecu de dif---------------#
salidas = [] #es la lista vacia
for nombre, tt, xx in entradas:
    y = sig.lfilter(b, a, xx) #me aplica la ecu en diferencias
    salidas.append((nombre, tt, xx, y)) #append me mete cosas en la lista
    print("#--- " + nombre + " ---#")
    print("\nPotencia salida =", potencia(y))
    print("\nEnergía salida =", energia(y))
    print("\n")


#---------------Grafico---------------#

#Senoidal 2 kHz
plt.figure(1)
plt.plot(t, x, label="Entrada", color="darkgray")
plt.plot(t, salidas[0][3], label="Salida", color="mediumvioletred")
#con salida[x][y] lo que hago es tomar el elemento (que es un struct) nro x de la lista
# y luego dentro de ese struct tomo el elemento nro y 

plt.title("Senoidal 2 kHz")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Senoidal amplificada y desfasada
plt.figure(2)
plt.plot(t1, x1, label="Entrada", color="darkgray")
plt.plot(t1, salidas[1][3], label="Salida", color="blueviolet")

plt.title("Senoidal amplificada y desfasada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Modulada
plt.figure(3)
plt.plot(t, modulada, label="Entrada", color="darkgray")
plt.plot(t, salidas[2][3], label="Salida", color="deeppink")

plt.title("Modulada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Recortada
plt.figure(4)
plt.plot(t, recortada, label="Entrada", color="darkgray")
plt.plot(t, salidas[3][3], label="Salida", color="mediumpurple")

plt.title("Recortada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Cuadrada
plt.figure(5)
plt.plot(tq, cuadrada, label="Entrada", color="darkgray")
plt.plot(tq, salidas[4][3], label="Salida", color="mediumorchid")

plt.title("Cuadrada")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Pulso
plt.figure(6)
plt.plot(tp, pulso, label="Entrada", color="darkgray")
plt.plot(tp, salidas[5][3], label="Salida", color="darkturquoise")

plt.title("Pulso")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#---------------respuesta al impulso h[n]---------------#
delta = np.zeros(N) 
delta[0] = 1
respuestaPulso= sig.lfilter(b, a, delta)  #salida del sistema ante el impulso

plt.figure(7)
plt.plot(respuestaPulso[:50], 'o', color='palevioletred')  #respuestaPulso[:50] poruqe solo un par de puntos y no graficar mucho
plt.title("Respuesta al impulso h[n]")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.grid(True)
plt.tight_layout()
plt.show()


#para generar la salida de una señal de entrada usando la respuesta al impulso del sistema,
#tengo que usar comvolucion entre la señal de entrada y la respuesta al impulso
#en lugar de usar directamente la función lfilter. Esto permite obtener la salida
# osea y[n]=x[n]*h[n]

#------------Con la primmer señal de todas------------#
conv = np.convolve(x, respuestaPulso)[:N]  #[:N] para tener el mismo tamaño

plt.figure(8)
plt.plot(t, x, label="Entrada", color="darkgray")
plt.plot(t, conv, label="Salida (convolucion)", color="purple")
plt.title("Salida usando la respuesta al impulso (Senoidal 2 kHz)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------Comprobación entre convolucion y lfilter------------#
plt.figure(12)
plt.plot(t, conv, label="Salida con convolucion", color="purple", linewidth=3)
plt.plot(t, salidas[0][3], label="Salida con lfilter", color="powderblue", linestyle='--')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

#--------------------Punto 2--------------------#


#--------------------A: y[n] = x[n] + 3 x[n-10]--------------------#
hA = np.zeros(11)
hA[0] = 1
hA[10] = 3 #esto esta asi porque como para la delta solo vale 1 en n=0, mi sistema tiene solo esos dos valores que puse
#la salida depende solo de la entrada, asi que al darle un pulso (delta) la respuesta al impulso es 
#exactamente los coeficientes de entrada


# salida con el sen
salidaA = np.convolve(x, hA)[:N]

plt.figure(9)
plt.subplot(2,1,1)
plt.plot(t, salidaA,color="salmon")
plt.title('Sistema A - Salida')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.tight_layout()
plt.grid(True)

plt.subplot(2,1,2) 
plt.plot(range(len(hA)), hA,'o', color='mediumvioletred')
plt.title('Sistema A - Respuesta al impulso')
plt.xlabel('n (muestras)')
plt.ylabel('hA[n]')
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Sistema B: y[n] = x[n] + 3 y[n-10] ---
aB = np.zeros(11)
aB[0] = 1
aB[10] = -3
bB = np.array([1]) #porque no hay nadaen x

#respuesta al impulso
delta = np.zeros(N)
delta[0] = 1
hB= sig.lfilter(bB, aB, delta)
#la salida depende tambien de salidas pasadas, así que al darle un pulso (delta) la respuesta al impulso
#sigue retroalimentándose y hay que calcularla con el lfilter

# salida para la senoidal (simulación)
hBcortada = hB[:30]
salidaB = np.convolve(x, hBcortada)[:N] #lo corte asi no se ve como explota todo
#las lineas de codigo comentadas son como estaba originalmente (graficaba una especie de linea que iba hacia arriba)
#salidaB = np.convolve(x, hB)[:N]

plt.figure(10)
plt.subplot(2,1,1)
plt.plot(t, salidaB, color="skyblue")
plt.title('Sistema B - Salida')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.subplot(2,1,2) 
plt.plot(hB, label='respuesta al impulso', color="blueviolet")
plt.title('Sistema B - Respuesta al impulso')
plt.xlabel('n (muestras)')
plt.ylabel('hB[n]')
plt.grid(True)
plt.tight_layout()
plt.show()

#Energia de las salidas
energiaA = energia(salidaA)
energiaB = energia(salidaB)

# Potencia de las salidas
potenciaA = potencia(salidaA)
potenciaB = potencia(salidaB)

print("Energia salida A:", energiaA)
print("Potencia salida A:", potenciaA)
print("Energia salida B:", energiaB)
print("Potencia salida B:", potenciaB)


# %%

#--------------------BONUS--------------------#

# Parámetros de la senoidal (flujo)
Qn = 20   # amplitud de la onda de flujo ml/s
dc = 80   # flujo medio equivalente a presión base en mmHg
ff = 1     #frecuencia Hz
ph = 0       #fase inicial
fs = 100     #frecuencia de muestreo Hz
C = 1.5     # ml/mmHg
R = 1.0     # mmHg·s/ml
dt = 1/fs   # paso temporal consistente con la señal

t, Q = sen(Qn, dc, ff, ph, N, fs)

P = np.zeros(N) #es un array donde almaceno la presion arterial en cada instante
P[0] = 80  # (valor típico de presión sistólica inicial o presión de referencia)

#metodo de Euler para discretizar la ecuacion diferencial
for n in range(N-1):#itero de la muestra 0 hasta la penúltima N-1
    P[n+1] = P[n] + dt*(Q[n] - P[n]/R)/C

#en cada paso la presion aumenta si el flujo Q[n] es mayor que P[n]/R, y disminuye si es menor

plt.figure(11)
plt.plot(t, Q, label='Flujo Q(t)', color='skyblue')
#Q es la entrada del sistema, simula el flujo sanguineo pulsatil (un latido por segundo)
plt.plot(t, P, label='Presión arterial P(t)', color='mediumvioletred')
#P es la salida, es la presion arterial que responde al flujo
plt.xlabel('Tiempo [s]')
plt.ylabel('Presion[mmHg] / Flujo[ml/s]')
plt.title('Modelo Windkessel - Discretizado con Euler')
plt.legend()
plt.grid(True)
plt.show()











