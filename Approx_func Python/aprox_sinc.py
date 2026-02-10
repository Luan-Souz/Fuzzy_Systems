###################         Universidade do Estado do Amazonas          ###################
###################         Tópicos Especiais 1: Lógica Fuzy            ###################
###################         Aproximador Fuzzy - Função Sinc             ###################
###################         Prof. Dr. Rodrigo Araújo                    ###################
###################         Aluno: Luan Souza                           ###################
#%%
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sympy import sinc, symbols, solve, N, pi

n = 3
x = np.linspace(0, np.pi, 1000)

# Funções de pertinência
sigma = 0.35

u = np.zeros((len(x), 7))
u[:, 0] = fuzz.gaussmf(x, 0, sigma)
u[:, 1] = fuzz.gaussmf(x, 0.75, sigma)
u[:, 2] = fuzz.gaussmf(x, 1.3, sigma)
u[:, 3] = fuzz.gaussmf(x, 1.95, sigma)
u[:, 4] = fuzz.gaussmf(x, 2.5, sigma)
u[:, 5] = fuzz.gaussmf(x, 2.85, sigma)
u[:, 6] = fuzz.gaussmf(x, np.pi, sigma)

# Coefficients calculation
#sinc_func = lambda x_val: float(sinc(x_val))

#a1, b1 = np.polyfit(np.array([0, 1.43]), np.array([sinc_func(0), sinc_func(1.43)]), 1)
#a2, b2 = np.polyfit(np.array([1.43, 2.46]), np.array([sinc_func(1.43), sinc_func(2.46)]), 1)
#a3, b3 = np.polyfit(np.array([2.46, np.pi]), np.array([sinc_func(2.46), sinc_func(np.pi)]), 1)

# Define os símbolos
a, b = symbols('a b')

# Primeira equação
eq1 = b - sinc(0)
eq2 = 1.43*a + b - sinc(1.43)
solutions = solve((eq1, eq2), (a, b))

a1 = N(solutions[a])
b1 = N(solutions[b])

# Segunda equação
eq1 = 1.43*a + b - sinc(1.43)
eq2 = 2.46*a + b - sinc(2.46)
solutions = solve((eq1, eq2), (a, b))

a2 = N(solutions[a])
b2 = N(solutions[b])

# Terceira equação
eq1 = 2.46*a + b - sinc(2.46)
eq2 = pi*a + b - sinc(pi)
solutions = solve((eq1, eq2), (a, b))

a3 = N(solutions[a])
b3 = N(solutions[b])

# Convertendo para float para resultados finais
a1 = float(a1)
b1 = float(b1)
a2 = float(a2)
b2 = float(b2)
a3 = float(a3)
b3 = float(b3)


plt.figure(figsize=(10, 8))

# Funções de pertinência
plt.subplot(n, 1, 1)
plt.grid(True)
for i in range(u.shape[1]):
    plt.plot(x, u[:, i])
plt.xlabel('x')
plt.ylabel('$\mu_i(x)$')
plt.legend(['$\mu_1(x)$', '$\mu_2(x)$', '$\mu_3(x)$', '$\mu_4(x)$', '$\mu_5(x)$', '$\mu_6(x)$', '$\mu_7(x)$'])
plt.xlim([0, np.pi])
plt.ylim([0, 1])

# Regras
y = np.zeros((len(x), 7))
y[:, 0] = np.ones_like(x)
y[:, 1] = a1 * x + b1
y[:, 2] = -0.2172 * np.ones_like(x)
y[:, 3] = a2 * x + b2
y[:, 4] = 0.1284 * np.ones_like(x)
y[:, 5] = a3 * x + b3
y[:, 6] = -0.0436 * np.ones_like(x)

# Aproximação fuzzy
y_fuzzy = np.sum(u * y, axis=1) / np.sum(u, axis=1)
y_real = np.sinc(x)

plt.figure()

# Aproximação fuzzy vs. real
plt.grid(True)
plt.plot(x, y_real, 'k', label='real')
plt.plot(x, y_fuzzy, 'r', label='estimada')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.xlim([0, np.pi])

# Erro e Erro Quadrático Médio (MSE)
plt.figure()
error = y_fuzzy - y_real
plt.grid(True)
plt.plot(x, error, label='erro de aproximação')
MSE = np.mean(error ** 2)
plt.plot(x, np.ones_like(x) * MSE, label='MSE')
plt.xlabel('x')
plt.ylabel('erro')
plt.legend()
plt.xlim([0, np.pi])

plt.show()

#%%
