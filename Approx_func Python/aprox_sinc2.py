###################         Universidade do Estado do Amazonas          ###################
###################         Tópicos Especiais 1: Lógica Fuzy            ###################
###################         Aproximador Fuzzy - Função Sinc             ###################
###################         Prof. Dr. Rodrigo Araújo                    ###################
###################         Aluno: Luan Souza                           ###################
#%%
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sympy import sinc, symbols, solve, diff, sin, N

n = 3
x = np.linspace(0, np.pi, 1000)

# Funções de pertinência
sigma = 0.4

u = np.zeros((len(x), 7))
u[:, 0] = fuzz.gaussmf(x, 0, sigma)
u[:, 1] = fuzz.gaussmf(x, 0.65, sigma)
u[:, 2] = fuzz.gaussmf(x, 1.3, sigma)
u[:, 3] = fuzz.gaussmf(x, 1.95, sigma)
u[:, 4] = fuzz.gaussmf(x, 2.5, sigma)
u[:, 5] = fuzz.gaussmf(x, 2.85, sigma)
u[:, 6] = fuzz.gaussmf(x, np.pi, sigma)

# Define the symbolic variable
a, b = symbols('a b')

# Derivada da função sinc
dy = diff(sinc(a), a)

# Resolver o sistema de equações para encontrar Sa e Sb
eq1 = b - sinc(0)
eq2 = 1.43*a + b - sinc(1.43)
solutions = solve((eq1, eq2), (a, b))

a1 = N(solutions[a])
b1 = N(solutions[b])

# Calcular os valores para a1, b1, a2, b2, a3, b3
a1 = dy.subs(a, 0.715)
b1 = sinc(0.715) - a1*0.715

a2 = dy.subs(a, 1.945)
b2 = sinc(1.945) - a2*1.945

a3 = dy.subs(a, 2.8008)
b3 = sinc(2.8008) - a3*2.8008

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
