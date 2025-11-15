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

#cantidad ded muestras de la mediana
cant_muestras_med = 201

ECG_med200 = signal.medfilt(ecg_one_lead, cant_muestras_med) #kernel size impar porque algo par no tiene media
#200 es la canti

ECG_med600 = signal.medfilt(ECG_med200, 601) #kernel size impar porque algo par no tiene media

resta = ecg_one_lead-ECG_med600

plt.figure(1)
plt.plot(ecg_one_lead, label = 'ecg', color = 'orchid')
plt.plot(ECG_med600, label = 'med 600', color = 'cornflowerblue')
plt.plot(resta, label = 'resta', color = 'rebeccapurple')
plt.title('Mediana')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('|H(jω)| [dB]')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.xlim(3000, 5000)


# %% cubib spline

fs = 1000
qrs_detections = mat_struct['qrs_detections'].flatten()
n0 = int(0.06 * fs) #numero random de ms antes del QRS
m_i = qrs_detections - n0 #posiciones en el tiempo

#valores de ECG en esos puntos
s_m = ecg_one_lead[m_i] #valores del ecg en las posiciones m_i

Cspline = interpolate.CubicSpline(m_i, s_m) #crea la función spline que interpola entre los puntos PQ
b = Cspline(np.arange(N)) #evalua la spline en todos los puntos del ECG

restaCS = ecg_one_lead-b

plt.figure(2)
plt.plot(ecg_one_lead, label = 'ecg', color = 'yellowgreen')
plt.plot(b, label='spline', color='pink')
#plt.plot(restaCS, label = 'resta', color = 'skyblue')
plt.title('Spline Cubico')
plt.legend()
#plt.xlim(3000, 5000)
plt.grid(True)
plt.show()

# %% comparacion
plt.figure(3)
plt.plot(resta, label = 'mediana', color = 'lightcoral')
plt.plot(restaCS, label = 'spline', color = 'mediumpurple')
plt.title('comparacion')
plt.legend()
plt.xlim(4250, 4500)
plt.grid(True)
plt.show()



# %% 3

qrs_real = mat_struct['qrs_detections'].flatten().astype(int)
patron = mat_struct['qrs_pattern1'].flatten().astype(float)
# Se invierte el patrón para hacer correlación

#patron escalado y con valor medio 0
patron = patron - np.mean(patron)
patron = patron / np.std(patron)
valor_medio_patron = np.mean(patron)#verificoo

ecg_detection = signal.lfilter(b=patron[::-1], a=1, x=restaCS.astype(float))
#en vez de usar ecg_one_lead, uso restaCS para la correlacion, esto elimina la línea base y mejora la coincidencia con el patrón

#valor absol
ecg_detection_abs = np.abs(ecg_detection)



#deteccion de picos

umbral = 0.3 * np.max(ecg_detection_abs)#me armo un umbral adaptativo para evitar picos debiles que no son qrs

mis_qrs, _ = signal.find_peaks(ecg_detection_abs,
                               height=umbral,
                               distance=200)

demora = len(patron) - 1
mis_qrs_correct = mis_qrs - demora

plt.figure(4)
plt.plot(ecg_one_lead, label="ECG original", color="orchid", linewidth=1)

# Picos detectados
plt.plot(mis_qrs, ecg_one_lead[mis_qrs], "go", label="QRS detectados", markersize=6)

# Picos reales
plt.plot(qrs_real, ecg_one_lead[qrs_real], "rx", label="QRS reales", markersize=6)

#
plt.plot(qrs_real)
plt.plot(ecg_detection_abs)

plt.title("Comparación de detecciones QRS")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.legend(loc="upper right")
plt.grid(True, linestyle=":")
#plt.xlim(8000, 20000)
plt.tight_layout()
plt.show()


def validar_detecciones(qrs_real, qrs_detectados, tolerancia=50):
    TP = 0  # Verdaderos positivos
    FP = 0  # Falsos positivos
    FN = 0  # Falsos negativos

    detectados_usados = np.zeros(len(qrs_detectados), dtype=bool)

    for qrs in qrs_real:
        # Buscar detección dentro de la tolerancia
        coincidencias = np.where(np.abs(qrs_detectados - qrs) <= tolerancia)[0]
        if len(coincidencias) > 0:
            idx = coincidencias[0]
            if not detectados_usados[idx]:
                TP += 1
                detectados_usados[idx] = True
            else:
                FN += 1
        else:
            FN += 1

    FP = np.sum(~detectados_usados)

    sensibilidad = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return sensibilidad, precision, TP, FP, FN

# Aplicación
sens, prec, TP, FP, FN = validar_detecciones(qrs_real, mis_qrs_correct)
print(f"Sensibilidad: {sens:.3f}, Precisión: {prec:.3f}")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")



fs = 1000
ventana = int(0.2 * fs)  # 200 ms alrededor del QRS
qrs_real = mat_struct['qrs_detections'].flatten().astype(int)

# Seleccionamos algunos QRS (por ejemplo, los primeros 20 que estén bien ubicados)
segmentos = []
for qrs in qrs_real[:20]:
    if qrs - ventana//2 >= 0 and qrs + ventana//2 < len(ecg_one_lead):
        segmento = ecg_one_lead[qrs - ventana//2 : qrs + ventana//2]
        segmentos.append(segmento)

segmentos = np.array(segmentos)


plt.figure(figsize=(10, 6))

# Graficamos todos los QRS reales
for i, seg in enumerate(segmentos):
    plt.plot(seg, alpha=0.4, label=f'QRS {i+1}' if i < 5 else None, color='gray')

# Graficamos el patrón
patron = mat_struct['qrs_pattern1'].flatten().astype(float)
patron = patron - np.mean(patron)
patron = patron / np.std(patron)
plt.plot(patron, label='Patrón QRS', color='red', linewidth=2)

plt.title('Comparación entre QRS reales y patrón')
plt.xlabel('Muestras')
plt.ylabel('Amplitud normalizada')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()









