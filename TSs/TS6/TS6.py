import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.patches as patches


#-----------FUncion hecha para calcular la repsuesta asintótica-----------#
def bode_asintotica(w, quiebres, nivel_baja_db):
    y = np.full_like(w, nivel_baja_db, dtype=float)
    qs = sorted(quiebres, key=lambda x: x[0])
    slope = 0.0
    last_w = w[0]
    last_y = nivel_baja_db
    qi = 0
    for i, wi in enumerate(w):
        while qi < len(qs) and wi >= qs[qi][0]:
            w_break, delta = qs[qi]
            if w_break > last_w:
                last_y += slope * np.log10(w_break/last_w)
                last_w = w_break
            slope += delta
            qi += 1
        y[i] = last_y + slope * np.log10(wi/last_w) if wi >= last_w else last_y
    return y


#-----------Coeficientes del numerador y denominador-----------#
# %% T1(s) = (s^2 + 9) / (s^2 + sqrt(2)s + 1)
num1 = [1, 0, 9]
den1 = [1, np.sqrt(2), 1]

w1, h1 = signal.freqs(num1, den1, worN=np.logspace(-1, 2, 1000))
phase1 = np.unwrap(np.angle(h1))
gd1 = -np.diff(phase1) / np.diff(w1)

plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.semilogx(w1, 20*np.log10(abs(h1)), label='T1(s)', color='lightblue')
asint1 = bode_asintotica(w1, [(1.0, -40.0), (3.0, +40.0)], 20*np.log10(9.0))
plt.semilogx(w1, asint1, '--', color='orchid', label='Asintótica')
plt.title('Magnitud - T1(s)')
plt.xlabel('ω [rad/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()


plt.subplot(2,1,2)
plt.semilogx(w1, np.degrees(phase1), label='T1(s)', color="orchid")
plt.title('Fase - T1(s)')
plt.xlabel('ω [rad/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()

# %% T2(s) = (s^2 + 1/9) / (s^2 + s/5 + 1)
num2 = [1, 0, 1/9]
den2 = [1, 1/5, 1]

w2, h2 = signal.freqs(num2, den2, worN=np.logspace(-1, 2, 1000))
phase2 = np.unwrap(np.angle(h2))
gd2 = -np.diff(phase2) / np.diff(w2)

plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.semilogx(w2, 20*np.log10(abs(h2)), label='T2(s)', color='lightblue')
asint2 = bode_asintotica(w2, [(1/3, +40.0), (1.0, -40.0)], 20*np.log10(1/9))
plt.semilogx(w2, asint2, '--', color='orchid', label='Asintótica')
plt.title('Magnitud - T2(s)')
plt.xlabel('ω [rad/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(2,1,2)
plt.semilogx(w2, np.degrees(phase2), label='T2(s)', color="orchid")
plt.title('Fase - T2(s)')
plt.xlabel('ω [rad/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()

# %% T3(s) = (s^2 + s/5 + 1) / (s^2 + sqrt(2)s + 1)
num3 = [1, 1/5, 1]
den3 = [1, np.sqrt(2), 1]

w3, h3 = signal.freqs(num3, den3, worN=np.logspace(-1, 2, 1000))
phase3 = np.unwrap(np.angle(h3))
gd3 = -np.diff(phase3) / np.diff(w3)

plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.semilogx(w3, 20*np.log10(abs(h3)), label='T3(s)', color='lightblue')
asint3 = np.zeros_like(w3)
plt.semilogx(w3, asint3, '--', color='orchid', label='Asintótica')
plt.title('Magnitud - T3(s)')
plt.xlabel('ω [rad/s]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.subplot(2,1,2)
plt.semilogx(w3, np.degrees(phase3), label='T3(s)', color="orchid")
plt.title('Fase - T3(s)')
plt.xlabel('ω [rad/s]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()
plt.tight_layout()
plt.show()

# %%

# Diagrama de polos y ceros

z1, p1, k1 = signal.tf2zpk(num1, den1)
z2, p2, k2 = signal.tf2zpk(num2, den2)
z3, p3, k3 = signal.tf2zpk(num3, den3)


plt.figure(figsize=(6,6))

# T1
plt.plot(np.real(p1), np.imag(p1), 'x', markersize=10, label='T1 Polos', color='orchid', markeredgewidth=2.5)
if len(z1) > 0:
    plt.plot(np.real(z1), np.imag(z1), 'o', markersize=10, fillstyle='none', label='T1 Ceros', color='lightblue', markeredgewidth=2.5)


# Círculo unitario
unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                             color='gray', ls='dotted', lw=2)


axes_hdl = plt.gca()
axes_hdl.add_patch(unit_circle)

plt.axis([-1.1, 1.1, -3.5, 3.5])
plt.axis('equal')
plt.title('T1 - Diagrama de Polos y Ceros (plano s)')
plt.xlabel(r'$\Re(s)$')
plt.ylabel(r'$\Im(s)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(6,6))
# T2
plt.plot(np.real(p2), np.imag(p2), 'x', markersize=10, label='T2 Polos', color='orchid', markeredgewidth=2.5)
if len(z2) > 0:
    plt.plot(np.real(z2), np.imag(z2), 'o', markersize=10, fillstyle='none', label='T2 Ceros', color='lightblue', markeredgewidth=2.5)

# Círculo unitario
unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                             color='gray', ls='dotted', lw=2)
axes_hd2 = plt.gca()
axes_hd2.add_patch(unit_circle)

plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title('T2 - Diagrama de Polos y Ceros (plano s)')
plt.xlabel(r'$\Re(s)$')
plt.ylabel(r'$\Im(s)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(6,6))
# T3
plt.plot(np.real(p3), np.imag(p3), 'x', markersize=10, label='T3 Polos', color='orchid', markeredgewidth=2.5)
if len(z3) > 0:
    plt.plot(np.real(z3), np.imag(z3), 'o', markersize=10, fillstyle='none', label='T3 Ceros', color='lightblue', markeredgewidth=2.5)
    

axes_hd3 = plt.gca()
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)

# Círculo unitario
unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                             color='gray', ls='dotted', lw=2)
axes_hd3.add_patch(unit_circle)

plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title('T3 - Diagrama de Polos y Ceros (plano s)')
plt.xlabel(r'$\Re(s)$')
plt.ylabel(r'$\Im(s)$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
