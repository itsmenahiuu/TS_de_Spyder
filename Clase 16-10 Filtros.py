import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Parametros del filtro


wp = 1 #frecuencia de corte/paso
ws = 3 #fr4cuecia de stop/detenida --- np.abs(p) da casi 1 si ws=3

alpha_p = 1 #atenuacion de corte- atenuacion maxima a la wp, alpha_max, perdidas en banda de paso                                       
alpha_s = 40 #atenuacion minima a la ws, alpha_min, minima atenuacion requerida
#en banda de paso


#f_aprox = 'butter'
f_aprox = 'cheby1'
# f_aprox = 'cheby2'
# f_aprox = 'ellip'
# f_aprox = 'cauer'


#Diseño del filtro butterworth analógico
b, a = signal.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, analog=True, ftype=f_aprox, output='ba')
# ba devuelve dos listas que son los coeficientes del polinomio osea de p y q
w,h = signal.freqs(b, a, worN=np.logspace(-1, 2, 1000)) #10Hz a 1MHz aprox
#w, h = signal.freqs(b, a) #calcula la respuesta en frecuencia del filtro

phase = np.unwrap(np.angle(h))
gd = -np.diff(phase) / np.diff(w) #retardo de grupo

# --- Polos y ceros ---
z, p, k = signal.tf2zpk(b, a)

# --- Gráficas ---
#plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.semilogx(w, 20*np.log10(abs(h)), label=f'{f_aprox}')
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.subplot(2,2,2)
plt.semilogx(w, np.degrees(phase), label=f'{f_aprox}')
plt.title('Fase')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(2,2,3)
plt.semilogx(w[:-1], gd, label=f'{f_aprox}')
plt.title('Retardo de Grupo')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('τg [s]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
sos =signal.tf2sos(b,a, analog = True)





