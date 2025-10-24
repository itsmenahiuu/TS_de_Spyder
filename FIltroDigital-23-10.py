import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio


##################
## ECG con ruido
##################

# para listar las variables que hay en el archivo
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_ruido = mat_struct['ecg_lead'].flatten()  # señal con ruido (1D)
N_ecg_ruido = len(ecg_ruido)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

# plt.figure(7)
# plt.plot(ecg_ruido, color='mediumvioletred')
# plt.title('ECG con ruido')
# plt.xlabel('Muestras [n]')
# plt.ylabel('Amplitud [V]')
# plt.grid()
# plt.show()



#Parametros del filtro

fs = 1000 #Hz

#plantilla
wp = [0.8, 35] #frecuencia de corte/paso
ws = [0.1, 40] #fr4cuecia de stop/detenida --- np.abs(p) da casi 1 si ws=3
#[fin de la banda de stop inferior, fin de la anda de stop superior]

alpha_p = 1 #atenuacion de corte- atenuacion maxima a la wp, alpha_max, perdidas en banda de paso                                       
alpha_s = 40 #atenuacion minima a la ws, alpha_min, minima atenuacion requerida
#en banda de paso

#---------------------------------------------------------------------------------------------------------------------------#
#Cauer
f_aprox = 'cauer'
mi_sos_cauer = signal.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs = fs)

w,h = signal.freqz_sos(mi_sos_cauer, worN=np.logspace(-2, 1.9, 1000), fs = fs) #10Hz a 1MHz aprox
#w, h = signal.freqz_sos(mi_sos, fs = fs) #calcula la respuesta en frecuencia del filtro

# --- Polos y ceros ---
z, p, k = signal.sos2zpk(mi_sos_cauer)


#---------------------------------------------------------------------------------------------------------------------------#
#cheby1
f_aprox = 'cheby1'
mi_sos_cheby1 = signal.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs = fs)

w,h = signal.freqz_sos(mi_sos_cheby1, worN=np.logspace(-2, 1.9, 1000), fs = fs) #10Hz a 1MHz aprox
#w, h = signal.freqz_sos(mi_sos, fs = fs) #calcula la respuesta en frecuencia del filtro

# --- Polos y ceros ---
z, p, k = signal.sos2zpk(mi_sos_cheby1)


#---------------------------------------------------------------------------------------------------------------------------#
#cheby2
f_aprox = 'cheby2'
mi_sos_cheby2 = signal.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs = fs)

w,h = signal.freqz_sos(mi_sos_cheby2, worN=np.logspace(-2, 1.9, 1000), fs = fs) #10Hz a 1MHz aprox
#w, h = signal.freqz_sos(mi_sos, fs = fs) #calcula la respuesta en frecuencia del filtro

# --- Polos y ceros ---
z, p, k = signal.sos2zpk(mi_sos_cheby2)


#---------------------------------------------------------------------------------------------------------------------------#
#butter
f_aprox = 'butter'
mi_sos_butter = signal.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs = fs)

w,h = signal.freqz_sos(mi_sos_butter, worN=np.logspace(-2, 1.9, 1000), fs = fs) #10Hz a 1MHz aprox
#w, h = signal.freqz_sos(mi_sos, fs = fs) #calcula la respuesta en frecuencia del filtro

# --- Polos y ceros ---
z, p, k = signal.sos2zpk(mi_sos_butter)

#---------------------------------------------------------------------------------------------------------------------------#

phase = np.unwrap(np.angle(h))
w_rad = w / (fs/2) *np.pi # paso a radianes porque el retardo de rgupo debe tener w en radianes asi da en muestras
gd = -np.diff(phase) / np.diff(w_rad) #retardo de grupo

# --- Gráficas ---
#plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(2,2,1)
plt.plot(w, 20*np.log10(abs(h)), label=f'{f_aprox}')
plt.title('Respuesta en Magnitud')
plt.xlabel('Pulsación angular [r/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.subplot(2,2,2)
plt.plot(w, np.degrees(phase), label=f'{f_aprox}')
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(2,2,3)
plt.plot(w[:-1], gd, label=f'{f_aprox}')
plt.title('Retardo de Grupo ')
plt.xlabel('Frecuencia [HZ]')
plt.ylabel('τg (# muestras)')
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
