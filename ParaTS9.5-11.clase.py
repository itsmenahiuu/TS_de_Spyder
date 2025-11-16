#filtros no lineales (no lti) (tiene que ver con la ts9)
# teoria de la estimacion, cuando tenes una variable aletoria y no podes ccalcular todo
#hay que hacer uso de un conocimiento previo de la ecg
#mediana, fitro de mediana, un peradore no linea que actua sobre una ventana de xms, la salida del sistema
#en una ventana de 200ms voy a tenr varlos muy altos o muy bajos
#si viene una transicion rapida la mediana no sigue el calculo rapido
#b es movimieno de linia de base, de baja freciencias, es la estimacion de las bajas frecuencias
#el de 200ms va a filtrar la qrs (nombre de parte de un ecg<)
#el filtro de 600ms va a filtrar ondas T o P
#en el segmento P-Q se ve el ruido (silencio electrico, dondd depuedo medir el rudio)
#vamos a asumir que le pegamos al P-Q (corremos el riesgo de caer en la onda P)
#como filtramos los 5 ciclos, muestreando 20ms, con filtro pasa bajo, 

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import interpolate
from scipy.signal import find_peaks

#ECG
mat_struct = sio.loadmat('./ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead'].flatten()
N = len(ecg_one_lead)
cant_muestras = N



#--------(1) Filtro de mediana--------#

fs = 1000
k200 = int(0.2 * fs) | 1   #cantidad de muestras de la mediana
k600 = int(0.6 * fs) | 1
#esta codeado de esta forma porque necesitaba asegurar que mi fs sea 1000Hz y que quede impar tmb

ECG_med200 = signal.medfilt(ecg_one_lead, k200) #kernel size impar porque algo par no tiene media
#200 es la canti

ECG_med600 = signal.medfilt(ECG_med200, k600) #kernel size impar porque algo par no tiene media

resta = ecg_one_lead-ECG_med600

plt.figure(1)
plt.plot(ecg_one_lead, label = 'ecg', color = 'orchid')
plt.plot(ECG_med600, label = 'med 600', color = 'cornflowerblue')
plt.plot(resta, label = 'resta', color = 'rebeccapurple')
plt.title('Mediana')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]") 
plt.legend()
plt.grid(True, which='both', ls=':')
plt.xlim(3000, 5000)


# %% cubib spline

#--------(2) Interpolación mediante splines cúbicos--------#

fs = 1000
qrs_detections = mat_struct['qrs_detections'].flatten()
n0 = int(0.06 * fs) #numero random de ms antes del QRS
m_i = qrs_detections - n0 #posiciones en el tiempo
m_i = m_i[m_i > 0]  #elimino los negativos para no tomar indices desde el final del vector

#valores de ECG en esos puntos
s_m = ecg_one_lead[m_i] #valores del ecg en las posiciones m_i

Cspline = interpolate.CubicSpline(m_i, s_m) #crea la función spline que interpola entre los puntos PQ
b = Cspline(np.arange(N)) #evalua la spline en todos los puntos del ECG

restaCS = ecg_one_lead-b

plt.figure(2)
plt.plot(ecg_one_lead, label = 'ecg', color = 'yellowgreen')
plt.plot(b, label='spline', color='pink')
plt.plot(restaCS, label = 'resta', color = 'skyblue')
plt.title('Spline Cubico')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]") 
plt.legend()
plt.xlim(3000, 5000)
plt.grid(True)
plt.show()

# %% comparacion (no es necesaria para el tp)
# plt.figure(3)
# plt.plot(resta, label = 'mediana', color = 'lightcoral')
# plt.plot(restaCS, label = 'spline', color = 'mediumpurple')
# plt.title('comparacion')
# plt.legend()
# plt.xlim(4250, 4500)
# plt.grid(True)
# plt.show()



# %% 3

#--------(3)  Filtro adaptado (matched filter)--------#



#----------------cosntruyo mi filtro adaptado----------------#
qrs_real = mat_struct['qrs_detections'].flatten().astype(int)
patron = mat_struct['qrs_pattern1'].flatten().astype(float)
# Se invierte el patrón para hacer correlación

#patron escalado y con valor medio 0
patron = patron - np.mean(patron)
patron = patron / np.std(patron)
valor_medio_patron = np.mean(patron)#verificoo


#normalizo
restaCS = restaCS - np.mean(restaCS)
restaCS = restaCS / np.std(restaCS)


#----------------aplico mi filtro adaptado----------------#
ecg_detection = signal.lfilter(b=patron[::-1], a=1, x=restaCS)
#en vez de usar ecg_one_lead, uso restaCS para la correlacion, esto elimina la línea base y mejora la coincidencia con el patrón

#El filtro adaptado es una correlación cruzada (una convolución con el patrón invertido)


#valor absol
ecg_detection_abs = np.abs(ecg_detection)
ecg_detection_abs = ecg_detection_abs / np.std(ecg_detection_abs)


#compenso mi demora
demora = len(patron) - np.argmax(patron)
ecg_detection_abs = np.roll(ecg_detection_abs, -demora)


#----------------deteccion de picos----------------#

umbral = 1

peaks, _ = signal.find_peaks(ecg_detection_abs, height=umbral, distance=300)

mis_qrs = peaks #mis detecciones


#----------------grafico----------------#

plt.figure(3)
plt.plot(restaCS, label="ECG filtrada (restaCS)", color='mediumpurple', zorder=1)
plt.scatter(qrs_real, restaCS[qrs_real], color='skyblue', label='QRS reales', s=25, zorder=2)
plt.scatter(mis_qrs, restaCS[mis_qrs], color='hotpink', label='QRS detectados',marker='*',s=25, zorder=3)
plt.title("Detección de latidos con filtro adaptado")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#zoom
plt.figure(4)
plt.plot(restaCS, label="ECG filtrada (restaCS)", color='mediumpurple', zorder=1)
plt.scatter(qrs_real, restaCS[qrs_real], color='skyblue', label='QRS reales', s=25, zorder=2)
plt.scatter(mis_qrs, restaCS[mis_qrs], color='hotpink', label='QRS detectados',marker='*',s=25, zorder=3)
plt.title("Detección de latidos (zoom)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.xlim(686000, 690000)
plt.ylim(-3, 4)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#----------------matriz de confusion----------------#

tol = int(0.05 * fs)   # tolerancia de 50 ms

TP = 0
FP = 0
FN = 0

qrs_real = np.array(qrs_real)
real_marcados = np.zeros(len(qrs_real), dtype=bool)

for p in mis_qrs:
    # buscar un QRS real cercano
    diffs = np.abs(qrs_real - p)
    idx = np.where(diffs < tol)[0]
    if len(idx) > 0:
        TP += 1
        real_marcados[idx[0]] = True
    else:
        FP += 1

FN = np.sum(~real_marcados)


#----------------metricas del detector----------------#

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
sensibilidad = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0
    
    
#----------------impresion de resultados----------------#

print("== MATRIZ DE CONFUSIÓN ==")
print("                Predicho")
print("               Sí      No")
print(f"Real Sí     [{TP:3d}    {FN:3d}]")
print(f"Real No     [{FP:3d}      - ]")

print("\n== MÉTRICAS ==")
print(f"Precisión:     {precision:.3f}")
print(f"Sensibilidad:  {sensibilidad:.3f}")
print(f"F1 Score:      {f1:.3f}")



