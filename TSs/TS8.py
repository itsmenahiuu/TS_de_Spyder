import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio

#----------Parametros y plantilla----------# punto a
fs = 1000
nyq_frec = fs / 2
ripple = 1       #(pasabanda)
atenuacion = 40   #(stopband)

#valores de la plantilla (decididos segUn TS59
ws1 = 0.1   # Hz  (stop baja)
wp1 = 0.8   # Hz  (pass baja)
wp2 = 35  # Hz  (pass alta)
ws2 = 40  # Hz  (stop alta)


frecs = np.array([0.0, ws1, wp1, wp2, ws2, nyq_frec]) / nyq_frec
gains_db = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])
gains = 10**(gains_db / 20)

print("Plantilla (Hz): ws1, wp1, wp2, ws2 =", ws1, wp1, wp2, ws2)

#----------Diseño de 2 filtros IIR (Butterworth y cauer)----------#

gpass = ripple/2  #divido por 2 porque voy a usar
gstop = atenuacion/2

#especifico wp y ws 
wp = (wp1, wp2)
ws = (ws1, ws2)

sos_butter = signal.iirdesign(wp=wp, ws=ws, gpass=gpass, gstop=gstop, ftype='butter', output='sos', fs=fs)
sos_cauer  = signal.iirdesign(wp=wp, ws=ws, gpass=gpass, gstop=gstop, ftype='cauer', output='sos', fs=fs)



#----------Diseño de 2 filtros FIR (ventana (firwin2) y Parks-McClellan (remez))----------#

#frecuencias para firwin2 en Hz 
freqs_fir = np.array([0.0, ws1, wp1, wp2, ws2, fs/2])

# respuesta deseada: 0 en stopbands y 1 en passband
desired_fir = np.array([0, 0, 1, 1, 0, 0])

#Cantidad de coeficientes
numtaps_win = 801   #impar para retardo entero (retardo = (numtaps-1)//2)
fir_win = signal.firwin2(numtaps=numtaps_win, freq=freqs_fir, gain=desired_fir, fs=fs, window='hann')

#Parks-McClellan (remez)
bands = [0, ws1, wp1, wp2, ws2, fs/2]
desired_remez = [0, 1, 0]
#weights
weights = [10, 1, 10]
numtaps_remez = 601
fir_remez = signal.remez(numtaps_remez, bands, desired_remez, weight=weights, fs=fs)

# ----------Respuestas en frecuencia----------#
w_iir, h_butter = signal.sosfreqz(sos_butter, worN=4096, fs=fs)
_, h_cauer = signal.sosfreqz(sos_cauer, worN=4096, fs=fs)
w_fir_win, h_fir_win = signal.freqz(fir_win, worN=4096, fs=fs)
w_fir_remez, h_fir_remez = signal.freqz(fir_remez, worN=4096, fs=fs)


plt.figure(1)
plt.semilogx(w_iir, 20*np.log10(np.abs(h_butter)+1e-12), label='IIR Butterworth')
plt.semilogx(w_iir, 20*np.log10(np.abs(h_cauer)+1e-12), label='IIR Cauer')
plt.semilogx(w_fir_win, 20*np.log10(np.abs(h_fir_win)+1e-12), label='FIR Window')
plt.semilogx(w_fir_remez, 20*np.log10(np.abs(h_fir_remez)+1e-12), label='FIR Remez')

#dibujo de la plantilla (zonas)
plt.axvspan(0, ws1, color='lightgray', alpha=0.25)
plt.axvspan(ws2, fs/2, color='lightgray', alpha=0.25)
plt.axvline(wp1, color='k', linestyle=':')
plt.axvline(wp2, color='k', linestyle=':')
# plt.xlim(wp1, wp2)
# plt.ylim([-80, 5])
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (dB)')
plt.title('Comparación respuestas en magnitud')
plt.legend()
plt.grid(True)
plt.show()



# ----------Cargo mi ecg y aplico los filtros----------#
mat_struct = sio.loadmat('./ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)


#Aplico IIR con filtfilt
ecg_iir_butter = signal.sosfiltfilt(sos_butter, ecg_one_lead)
ecg_iir_cauer  = signal.sosfiltfilt(sos_cauer, ecg_one_lead)


# Aplico FIR y compenso retardo para mostrar en plots
ecg_fir_win = signal.lfilter(fir_win, 1, ecg_one_lead)
delay_win = (len(fir_win)-1)//2
ecg_fir_remez = signal.lfilter(fir_remez, 1, ecg_one_lead)
delay_remez = (len(fir_remez)-1)//2


# Para poderlo ver en el grafico, tuve que correr la señal filtrada por el retardo
ecg_fir_win_corr = np.roll(ecg_fir_win, -delay_win)
ecg_fir_remez_corr = np.roll(ecg_fir_remez, -delay_remez)


# ----------Grafico----------#
plt.figure(2)
t = np.arange(0, 2000) / fs  # primeras 2 segundos (2000 muestras)
plt.plot(t, ecg_one_lead[:2000], label='ECG original', alpha=0.7)
plt.plot(t, ecg_iir_butter[:2000], label='IIR Butter (filtfilt)', linewidth=1)
plt.plot(t, ecg_fir_win_corr[:2000], label='FIR Window (compensado)', linewidth=1)
plt.legend()
plt.xlabel('Tiempo (s)')
plt.title('Comparación de señales filtradas (primeros 2 s)')
plt.grid()
plt.show()

# ----------evaluación en regiones (con y sin ruido)----------#
cant_muestras = N

# regiones con ruido
regs_ruido = (
    (4000, 5500),
    (10000, 11000),
)

for ii in regs_ruido:
    a = max(0, int(ii[0]))
    b = min(cant_muestras, int(ii[1]))
    rng = np.arange(a, b)

    plt.figure()
    plt.plot(rng, ecg_one_lead[rng], label='ECG raw', linewidth=1)
    plt.plot(rng, ecg_iir_butter[rng], label='IIR Butter', linewidth=1)
    plt.plot(rng, ecg_fir_win_corr[rng], label='FIR Window', linewidth=1)
    plt.title(f'Región con ruido: {a}-{b}')
    plt.legend()
    plt.gca().set_yticks(())
    plt.show()

# regiones sin ruido
regs_sin_ruido = (
    (int(5*60*fs), int(5.2*60*fs)),
    (int(12*60*fs), int(12.4*60*fs)),
)

for ii in regs_sin_ruido:
    a = max(0, int(ii[0]))
    b = min(cant_muestras, int(ii[1]))
    rng = np.arange(a, b)

    plt.figure()
    plt.plot(rng, ecg_one_lead[rng], label='ECG raw', linewidth=1)
    plt.plot(rng, ecg_iir_butter[rng], label='IIR Butter', linewidth=1)
    plt.plot(rng, ecg_fir_win_corr[rng], label='FIR Window', linewidth=1)
    plt.title(f'Región sin ruido: {a}-{b}')
    plt.legend()
    plt.gca().set_yticks(())
    plt.show()
