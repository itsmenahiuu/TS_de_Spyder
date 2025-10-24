import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io as sio

#Plantilla de diseño

# wp = 1 #frecuencia de corte (rad/s)
# ws = 5 #frecuencia de stop (rad/s)

# alpha_p = 3 #atenuación de corte (db), alpha maxima en bp
# alpha_s = 40 #atenuación de stop (db), alpha minima en bs

# f_aprox="butter"
# #f_aprox="cheby1"
# #f_aprox="cheby2"
# #f_aprox="cauer"

# #Filtro analogico

# b, a = signal.iirdesign(wp=wp, ws= ws, gpass=alpha_p, gstop=alpha_s, analog=True, ftype=f_aprox, output='ba') #devuelve los coeficientes de P y Q
# #np.logspace(1,6,1000) = espacio logaritmicamente espaciado con 1000 espacios entre 10 a la 1 y 10 a la 6

# w, h = signal.freqs(b, a, worN=np.logspace(-1,2,1000)) #calcula respuesta en frecuencia del filtro con h complejo

# phase = np.unwrap(np.angle(h)) #con unwrap evita la discontinuidad de fase (discontinuidad evitable)

# gd = -np.diff(phase)/np.diff(w) #retardo de grupo

# z, p, k = signal.tf2zpk(b, a) #convertimos a polos y ceros, z no deberia tener pq b es cte

# # --- Gráficas ---
# #plt.figure(figsize=(12,10))

# # Magnitud
# plt.subplot(2,2,1)
# plt.semilogx(w, 20*np.log10(abs(h)), label = f_aprox)
# plt.title('Respuesta en Magnitud')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('|H(jω)| [dB]')
# plt.grid(True, which='both', ls=':')

# # Fase
# plt.subplot(2,2,2)
# plt.semilogx(w, np.degrees(phase), label = f_aprox)
# plt.title('Fase')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('Fase [°]')
# plt.grid(True, which='both', ls=':')

# # Retardo de grupo
# plt.subplot(2,2,3)
# plt.semilogx(w[:-1], gd, label = f_aprox)
# plt.title('Retardo de Grupo')
# plt.xlabel('Pulsación angular [r/s]')
# plt.ylabel('τg [s]')
# plt.grid(True, which='both', ls=':')

# # Diagrama de polos y ceros
# plt.subplot(2,2,4)
# plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox}Polos')
# if len(z) > 0:
#     plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox}Ceros')
# plt.axhline(0, color='k', lw=0.5)
# plt.axvline(0, color='k', lw=0.5)
# plt.title('Diagrama de Polos y Ceros (plano s)')
# plt.xlabel('σ [rad/s]')
# plt.ylabel('jω [rad/s]')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# %%

#sos = signal.tf2sos(b, a, analog = True)

#%%

#Filtro digital  usando valores del ECG

fs = 1000 #Hz
wp = [0.8, 35] #frecuencia de corte (Hz)
ws = [0.1, 40] #frecuencia de stop (Hz) arriba de los 35 debe ser la segunda
#queremos q saque toda la frecuencia menor a 1 Hz (0.8)

alpha_p = 1 #atenuación de corte (db), alpha maxima en bp
alpha_s = 40 #atenuación de stop (db), alpha minima en bs

f_aprox="butter"
#f_aprox="cheby1"
#f_aprox="cheby2"
#f_aprox="cauer"

mi_sos = signal.iirdesign(wp=wp, ws= ws, gpass=alpha_p, gstop=alpha_s, analog=False, ftype=f_aprox, output='sos', fs = fs) #tiene polinomios de orden 2 (b0,b1,b2,a0,a1,a2) y de ornde de filtro N/2=38 (74 de orden)
#np.logspace(1,6,1000) = espacio logaritmicamente espaciado con 1000 espacios entre 10 a la 1 y 10 a la 6

w, h = signal.freqz_sos(mi_sos, worN=np.logspace(-2,1.9,1000), fs = fs)

phase = np.unwrap(np.angle(h)) #con unwrap evita la discontinuidad de fase (discontinuidad evitable)

w_rad = w / (fs/2) *np.pi
gd = -np.diff(phase)/np.diff(w_rad) #retardo de grupo

z, p, k = signal.sos2zpk(mi_sos)

# Magnitud
plt.subplot(2,2,1)
plt.plot(w, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')

# Fase
plt.subplot(2,2,2)
plt.plot(w, (phase), label = f_aprox)
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')

# Retardo de grupo
plt.subplot(2,2,3)
plt.plot(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [#muestras]')
plt.grid(True, which='both', ls=':')

# Diagrama de polos y ceros
plt.subplot(2,2,4)
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox}Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox}Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel('σ [rad/s]')
plt.ylabel('jω [rad/s]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


#%%

#Divido por 2 a los alfas pq con el filtfilt se va al doble de ripple sino (y doble de atenuación que es bueno eso) y asi podemos neutralizar la fase tranquilamente
mi_sos_butt = signal.iirdesign(wp=wp, ws= ws, gpass=alpha_p/2, gstop=alpha_s/2, analog=False, ftype="butter", output='sos', fs = fs)
mi_sos_cheby1 = signal.iirdesign(wp=wp, ws= ws, gpass=alpha_p/2, gstop=alpha_s/2, analog=False, ftype="cheby1", output='sos', fs = fs)
mi_sos_cheby2 = signal.iirdesign(wp=wp, ws= ws, gpass=alpha_p/2, gstop=alpha_s/2, analog=False, ftype="cheby2", output='sos', fs = fs)
mi_sos_cauer = signal.iirdesign(wp=wp, ws= ws, gpass=alpha_p/2, gstop=alpha_s/2, analog=False, ftype="cauer", output='sos', fs = fs)


#ECG
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)


ecg_filt_butt = signal.sosfiltfilt(mi_sos_butt, ecg_one_lead)
ecg_filt_cheby1 = signal.sosfiltfilt(mi_sos_cheby1, ecg_one_lead)
ecg_filt_cheby2 = signal.sosfiltfilt(mi_sos_cheby2, ecg_one_lead)
ecg_filt_cauer = signal.sosfiltfilt(mi_sos_cauer, ecg_one_lead)

plt.figure()
plt.plot(ecg_one_lead[:100000], label = 'ECG')
plt.plot(ecg_filt_butt[:100000], label = 'butter')
plt.plot(ecg_filt_cauer[:100000], label = 'cauer')
plt.plot(ecg_filt_cheby1[:100000], label = 'cheby1')
plt.plot(ecg_filt_cheby2[:100000], label = 'cheby2')
plt.legend()
