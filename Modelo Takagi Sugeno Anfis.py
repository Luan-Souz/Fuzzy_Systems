###################         Universidade do Estado do Amazonas          ###################
###################         Tópicos Especiais 1: Lógica Fuzy            ###################
###################         Gradiente Descendente para estimação        ###################
###################         de parâmetros de Modelo Fuzzy               ###################
###################         Prof. Dr. Rodrigo Araújo                    ###################
###################         Aluno: Luan Souza                           ###################

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

#%% Modelo fuzzy Takagi-Sugeno

def model(x, c, s, p, q):

    '''
        X: Random training data
        C: Center of membership functions
        S: Variance of membership functions
        P: Weight of each input variable
        Q: Polarization parameter

    This function estimates the fuzzy model parameters:

        yi_est: Estimate output of model
        yk: Output of the k-th rule
        wk: Activation degree of k-th rule      
        
    '''
    d = c.shape[0]
    m = c.shape[1]
    uk = np.zeros((d, m))
    yk = np.zeros(m)
    wk = np.zeros(m)


    # Calcula o grau de pertinência (membership degree)

    for k in range(m):
        for l in range(d):
            uk[l, k] = fuzz.gaussmf(x[l], c[l, k], s[l, k])

        # Calcula o output dos k-ésimos rules
        yk[k] = np.dot(p[:, k], x) + q[0,k]

        # Calcula o grau de ativação do k-ésimo rule
        wk[k] = np.prod(uk[:, k])

    # Calcula o output estimado do modelo
    yi_est = np.sum(wk * yk.T) / np.sum(wk)

    return yi_est, yk, wk
#%% Definindo parâmetros iniciais

alpha = 0.05   # taxa de aprendizado
m = 5          # número de regras
d = 1          # número de antecedentes
c = 2 * np.pi * np.random.rand(d, m)  # centros das funções de pertinência
s = np.random.rand(d, m)              # variâncias das funções de pertinência
a, b = -1, 1                           # Intervalo para os valores das variáveis de entrada
p = np.random.uniform(a, b, (d, m))    # peso de cada variável de entrada
q = np.random.uniform(a, b, (1, m))    # parâmetro de polarização

#%% Treinamento do modelo fuzzy

ni = 0
n = 100                               # número de dados de treinamento
data_train = 2 * np.pi * np.random.rand(n, 1)  # dados de treinamento
aux = np.random.uniform(a, b, (4, m))  # parâmetros auxiliares para critério de parada

while True:
    x = data_train[np.random.randint(n)]  # dados aleatórios de treinamento
    y = np.sinc(x)                         # saída desejada

    # Estimativa do modelo fuzzy
    yi_est, yk, wk = model(x, c, s, p, q)

    # Inicializando as matrizes bidimensionais
    dJdc = np.zeros((d, m))
    dJds = np.zeros((d, m))
    dJdp = np.zeros((d, m))
    dJdq = np.zeros((1, m))

    for k in range(m):
        dJdq[0, k] = -2 * (y - yi_est) * wk[k] / np.sum(wk)

        # Atualização do parâmetro (q)
        q[0, k] = q[0, k] - alpha * dJdq[0, k]

        for l in range(d):
            dJdc[l, k] = dJdq[0, k] * (yk[k] - yi_est) * (x[l] - c[l, k]) / s[l, k] ** 2
            dJds[l, k] = dJdq[0, k] * (yk[k] - yi_est) * (x[l] - c[l, k]) ** 2 / s[l, k] ** 3
            dJdp[l, k] = dJdq[0, k] * x[l]

            # Atualização dos parâmetros (c, s, p)
            c[l, k] = c[l, k] - alpha * dJdc[l, k]
            s[l, k] = s[l, k] - alpha * dJds[l, k]
            p[l, k] = p[l, k] - alpha * dJdp[l, k]


    ni += 1

    if np.linalg.norm(np.vstack((dJdc, dJds, dJdp, dJdq)) - aux) < 1e-6:
        break
    else:
        aux = np.vstack((dJdc, dJds, dJdp, dJdq))

#%% Estimativa e plotagem da função

x = np.arange(0, 2*np.pi, 0.01)
yi_est = np.zeros_like(x)
for i in range(len(x)):
    # Foi preciso usar o reshape no x[i] pois a função 'model' nao reconhece escalar
    yi_est[i], _, _ = model(np.reshape(x[i],newshape=(-1,1)), c, s, p, q)

y = np.sinc(x)

plt.plot(x, y, 'k', linewidth=2)
plt.plot(x, yi_est, 'r', linewidth=2)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$f(x)$', fontsize=16)
plt.legend(['Function', 'Approximation'])
plt.grid(True)
plt.xlim([0, 2*np.pi])
plt.show()

#%% Root Mean Squared Error (RMSE)

error = yi_est - y
RMSE = np.sqrt(np.mean(error**2))
print("RMSE:", RMSE)

#%% Plotando erro real

plt.hlines(0, 0, 2*np.pi, linestyles='dashed')
plt.plot(x, (yi_est-y), 'b', linewidth=2)
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$erro $', fontsize=16)
plt.legend(['Referência','Erro'])
plt.grid(True)
plt.xlim([0, 2*np.pi])
plt.show()
