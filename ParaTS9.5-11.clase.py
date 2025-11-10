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
# plt.title('Respuesta en Magnitud')
# plt.xlabel('Frecuencia [Hz]')
# plt.ylabel('|H(jω)| [dB]')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.xlim(3000, 5000)









