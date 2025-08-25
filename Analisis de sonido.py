import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fs, data = wavfile.read("coolGuitarra.wav") 

print("\nFrecuencia de muestreo:", fs, "Hz")
    
t = np.arange(len(data)) / fs #armo mi eje del tiempo para poder graficar

energiaIzq = np.sum(data[:, 0].astype(float)**2)
energiaDer = np.sum(data[:, 1].astype(float)**2)
#data[] agarra todas las filas de la columna 0 (osea canal izquierdo)
#astype(float) pasa de entero a float para evitar los problemas que salen cuandoelevo al cuadrado
#**2 eleva al cuadrado cada muestra
# np.sum() suma todos los valores al cuadrado

print("\nEnergia canal izquierdo:", energiaIzq)
print("\nEnergia canal derecho:", energiaDer)

#--------------------NORMALIZO--------------------#
data_float = data.astype(float) #convierto a float para evitar overflow y luego normalizo
max_val = np.max(np.abs(data_float))  #valor máximo absoluto de toda la señal
data_norm = data_float / max_val      #normalizo al rango [-1, 1]

# --- Energía normalizada por canal ---
energiaDer_norm = np.sum(data_norm[:, 1]**2)


print("Energía del audio:", energiaDer_norm)

#Yo ya se de antemano que mi sonido esta en dos canales (stereo) porque me lo avisaba la pagina de donde descargue dicho sonido
#Hago dos graficos superpuestos para poder ver ambos
plt.figure(1)

plt.subplot(2, 1, 1)
plt.plot(t, data[:, 0], color='plum')#data[] me elije la columna
#son dos columnas porque son dos channels
plt.title("Canal izquierdo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()


plt.subplot(2, 1, 2)
plt.plot(t, data[:, 1], color='teal')
plt.title("Canal derecho")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid()

plt.tight_layout()
plt.show()

