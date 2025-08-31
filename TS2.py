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
plt.ylabel("Amplitud")
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
plt.ylabel("Amplitud")
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
plt.ylabel("Amplitud")
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
plt.ylabel("Amplitud")
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
plt.ylabel("Amplitud")
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



#--------------------Punto 2--------------------#
print("\n--- Resultados punto 2 ---")
f = 2000
Omega = 2*np.pi*f/fs  # frecuencia digital
print(f"Frecuencia analizada: f={f} Hz, Omega={Omega:.4f} rad")

# (a) y[n] = x[n] + 3x[n-10]
Ha = 1 + 3*np.exp(-1j*10*Omega)
mag_a, phase_a = np.abs(Ha), np.angle(Ha)
print(f"(a) FIR: |H|={mag_a:.4f}, ∠H={phase_a:.4f} rad")

# (b) y[n] = x[n] + 3y[n-10]
Hb = 1 / (1 - 3*np.exp(-1j*10*Omega))
mag_b, phase_b = np.abs(Hb), np.angle(Hb)
print(f"(b) IIR: |H|={mag_b:.4f}, ∠H={phase_b:.4f} rad")
print("Nota: este sistema es inestable (|3|>1).")

plt.show()

