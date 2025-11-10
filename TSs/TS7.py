import numpy as np
import matplotlib.pyplot as plt

#malla de frecuencia
w = np.linspace(-np.pi, np.pi, 2000)
z = np.exp(-1j * w)

#coeficientes de mi FIR
b_a = np.array([1,1,1,1])       # a) y(n)=x(n−3)+x(n−2)+x(n−1)+x(n) 
b_b = np.array([1,1,1,1,1])     # b) x(n−4)+x(n−3)+x(n−2)+x(n−1)+x(n) 
b_c = np.array([1,-1])          # c) y(n)=x(n)−x(n−1) 
b_d = np.array([1,0,-1])        # d) y(n)=x(n)−x(n−2) 


# H(e^{jω})
H_a = np.polyval(b_a[::-1], z) #np.polyval evalua el polinomio en z = e^{-jω}
H_b = np.polyval(b_b[::-1], z)
H_c = np.polyval(b_c[::-1], z)
H_d = np.polyval(b_d[::-1], z)

#fases
fase_a = np.unwrap(np.angle(H_a))
fase_b = np.unwrap(np.angle(H_b))
fase_c = np.unwrap(np.angle(H_c))
fase_d = np.unwrap(np.angle(H_d))

# =======================================================
# a)

plt.figure(1)
plt.title("a) Módulo")
plt.plot(w, np.abs(H_a), color = "mediumorchid")
plt.ylabel("|H|")
plt.grid(True)

plt.subplot(2,1,2)
plt.title("a) Fase")
plt.plot(w, fase_a, color = "mediumorchid")
plt.xlabel("ω [rad]")
plt.ylabel("Fase [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()

# =======================================================
# b) 

plt.figure(2)

plt.subplot(2,1,1)
plt.title("b) Módulo")
plt.plot(w, np.abs(H_b), color = "mediumorchid")
plt.ylabel("|H|")
plt.grid(True)

plt.subplot(2,1,2)
plt.title("b) Fase")
plt.plot(w, fase_b, color = "mediumorchid")
plt.xlabel("ω [rad]")
plt.ylabel("Fase [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()

# =======================================================
# c)

plt.figure(3)
plt.subplot(2,1,1)
plt.title("c) Módulo")
plt.plot(w, np.abs(H_c), color = "mediumorchid")
plt.ylabel("|H|")
plt.grid(True)

plt.subplot(2,1,2)
plt.title("c) Fase")
plt.plot(w, fase_c, color = "mediumorchid")
plt.xlabel("ω [rad]")
plt.ylabel("Fase [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()

# =======================================================
# d)

plt.figure(4)
plt.subplot(2,1,1)
plt.title("d) Módulo")
plt.plot(w, np.abs(H_d), color = "mediumorchid")
plt.ylabel("|H|")
plt.grid(True)

plt.subplot(2,1,2)
plt.title("d) Fase")
plt.plot(w, fase_d, color = "mediumorchid")
plt.xlabel("ω [rad]")
plt.ylabel("Fase [rad]")
plt.grid(True)

plt.tight_layout()
plt.show()




