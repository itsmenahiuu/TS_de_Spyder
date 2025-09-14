import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal as sig
win = sig.windows


N=1000
fs=N

def SNRvarianza(señal, SNRdb):
    # Potencia de la señal
    potSeñal = np.mean(señal**2)

    # Varianza del ruido
    Rvar = potSeñal / (10**(SNRdb / 10))

    return Rvar

def sen( vmax, dc, ff, ph, nn, fs): 

    Ts = 1/fs 
    tt = np.linspace(0, (nn-1)*Ts, nn) 

    xx=vmax*np.sin(2*np.pi*ff*tt+ph)+dc 

    return tt,xx

t = np.arange(0, N)*(1/fs)

t, señal = sen(np.sqrt(2), 0, fs/4, 0, N, fs)

Rvar = SNRvarianza(señal, 10)
ruido = np.random.normal(0,np.sqrt(Rvar),N)

salida = señal + ruido
# varx = np.var(señal)
# varSalida = np.var(salida)
# print("Varianza: ", Rvar)
# print("Varianza: ", varx)
# print("Varianza: ", varSalida)

SEÑALfft = fft(señal)
SALIDAfft = fft(salida)


plt.figure()
plt.plot(10*np.log10(2*np.abs(SALIDAfft*(1/N))**2), label = "Señal + Ruido", color = 'mediumvioletred')
#plt.plot(20*np.log10(np.abs(SEÑALfft)), color = 'pink', label = "Señal")
plt.xlabel('Muestras [rad]')
plt.ylabel('Amplitud [dB]')
plt.title('Señal con SNR')
plt.legend()
plt.grid(True)
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.signal as sig
win = sig.windows


N=1000
R=200
fs=N
frecRandom= np.random.uniform(low=-2, high=2, size=R) #ojo que es un vector
frecRandom=frecRandom.reshape((N,R))

def SNRvarianza(señal, SNRdb):
    # Potencia de la señal
    potSeñal = np.mean(señal**2)

    # Varianza del ruido
    Rvar = potSeñal / (10**(SNRdb / 10))

    return Rvar

def sen( vmax, dc, ff, ph, nn, fs): 

    Ts = 1/fs 
    tt = np.linspace(0, (nn-1)*Ts, nn) 

    xx=vmax*np.sin(2*np.pi*ff*tt+ph)+dc 

    return tt,xx

t = np.arange(0, N)*(1/fs)


#quiero una matriz
t, señal = sen(np.sqrt(2), 0, fs/4, 0, N, fs)
señales = np.tile(señal, reps=(N, R))   

Rvar = SNRvarianza(señal, 10)
ruido = np.random.normal(0,np.sqrt(Rvar),size=(N, R))

salida = señal + ruido
# varx = np.var(señal)
# varSalida = np.var(salida)
# print("Varianza: ", Rvar)
# print("Varianza: ", varx)
# print("Varianza: ", varSalida)

SEÑALfft = fft(señal)
SALIDAfft = fft(salida, axis=0) #axis=0 es filas


plt.figure()
plt.plot(10*np.log10(2*np.abs(SALIDAfft*(1/N))**2), label = "Señal + Ruido", color = 'mediumvioletred')
#plt.plot(20*np.log10(np.abs(SEÑALfft)), color = 'pink', label = "Señal")
plt.xlabel('Muestras [rad]')
plt.ylabel('Amplitud [dB]')
plt.title('Señal con SNR')
plt.legend()
plt.grid(True)
plt.show()



# %%
#hecho por augusto
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import scipy.signal as sp


def gen_señal (fs, N, amp, frec, fase, v_medio, SNR):
    
    t_final = N * 1/fs
    tt = np.arange (0, t_final, 1/fs)
    
    frec_rand = np.random.uniform (-2, 2)
    frec_omega = frec/4 + frec_rand * (frec/N)
    
    ruido = np.zeros (N)
    for k in np.arange (0, N, 1):
        pot_snr = amp**2 / (2*10**(SNR/10))                                 
        ruido[k] = np.random.normal (0, pot_snr)
    
    x = amp * np.sin (frec_omega * tt) + ruido
    
    return tt, x

def eje_temporal (N, fs):
    
    Ts = 1/fs
    t_final = N * Ts
    tt = np.arange (0, t_final, Ts)
    return tt


def func_senoidal (tt, frec, amp, fase = 0, v_medio = 0):
    
    xx = amp * np.sin (2 * np.pi * frec * tt + fase) + v_medio # tt es un vector, por ende la función sin se evalúa para cada punto del mismo
    # xx tendrá la misma dimensión que tt
    return xx

SNR = 10 # SNR en dB
amp_0 = np.sqrt(2) # amplitud en V
N = 1000
fs = 1000
df = fs / N # Hz, resolución espectral

nn = np.arange (N) # vector adimensional de muestras
ff = np.arange (N) * df # vector en frecuencia al escalar las muestras por la resolución espectral
tt = eje_temporal (N = N, fs = fs)

s_1 = func_senoidal (tt = tt, amp = amp_0, frec = fs/4)

pot_ruido = amp_0**2 / (2*10**(SNR/10))        
#print (f"Potencia de SNR {pot_snr:3.1f}")   
                      
ruido = np.random.normal (0, np.sqrt(pot_ruido), N)
var_ruido = np.var (ruido)
print (f"Potencia de ruido -> {var_ruido:3.3f}")

x_1 = s_1 + ruido # modelo de señal

R = fft (ruido)
S_1 = fft (s_1)
X_1ruido = fft (x_1)
# print (np.var(x_1))


plt.plot (ff, 10*np.log10(np.abs(X_1ruido)**2), color='orange', label='X_1')
#plt.plot (ff, 20*np.log10(np.abs(S_1)), color='black', label='S_1')
#plt.plot (ff, 20*np.log10(np.abs(R)), label='Ruido')
plt.grid (True)
plt.legend ()
plt.show ()


# %%


