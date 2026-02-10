###################         Universidade do Estado do Amazonas          ###################
###################         Tópicos Especiais 1: Lógica Fuzy            ###################
###################         Segmentação de Imagem com C-Means           ###################
###################         Prof. Dr. Rodrigo Araújo                    ###################
###################         Aluno: Luan Souza                           ###################


import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skfuzzy as fuzz

#%% Função de normalizar imagem
def normalize_image(x):
    return x.astype(float) / 255

#%% Função objetivo fun
def fun(center, U, data):
    c = center.shape[0]  # number of clusters
    n = data.shape[0]    # number of data points
    y = 0
    for i in range(c):
        for j in range(n):
            y += U[i, j] * np.linalg.norm(center[i, :] - data[j, :]) ** 2
    return y

#%% Função CMeans
def cmeans(data, c, m, e):
    n = data.shape[0]                          # numero de observações
    center = np.random.rand(c, data.shape[1])  # iniciar centros aleatoriamente
    k = 1                                      # contador de iterações
    obj_fun = []                               # lista dos resultados da funcao objetivo
    C= np.zeros_like(center)

    while True:
        # Atualizando a matriz de pertinência

        U = np.zeros((c, n))
        for j in range(n):
            for i in range(c):
                dist = np.linalg.norm(center[i, :] - data[j, :])
                if dist == 0:
                    U[i, j] = 1
                    break
                else:
                    s_d = 0
                    for l in range(c):
                        s_d += (dist / np.linalg.norm(center[l, :] - data[j, :])) ** (2 / (m - 1))
                    U[i, j] = 1 / s_d

        # Calculando o centro dos novos clusters

        for i in range(c):
            s_ux = np.zeros(data.shape[1])
            s_u = 0
            for j in range(n):
                s_ux += (U[i, j] ** m) * data[j, :]
                s_u += U[i, j] ** m
            C[i, :] = s_ux / s_u

        # Calculando erro

        aux = 0
        for i in range(c):
            if np.linalg.norm(C[i, :] - center[i, :]) > e:
                aux = 1
                break

        # Atualizando centro dos clusters

        center[0:c, :] = C[0:c, :]

        # Calculando a função objetivo na k-ésima iteração

        obj_fun_k = fun(center, np.power(U, m), data)
        obj_fun.append(obj_fun_k)
        print('Iteration count = {}, obj.fcn = {}'.format(k, obj_fun_k))

        if aux == 0:
            break
        else:
            k += 1

    return center, U, obj_fun, k

#%% Segmentação de Imagem

plt.figure()
img = io.imread('img02.jpg')
plt.imshow(img)
plt.title('Original Image')

data = normalize_image(img.reshape(-1, 3))

alg = 1  # 0: função da lib skfuzzy; 1: função implementada
e = 1e-3 # erro máximo

for c in range(5, 6):  # número de clusters
    m = 2  # "poder" da funcao de pertinência

    if alg ==0:
        # Usando a funçào cmeans da biblioteca skfuzzy
        cntr, U, _, _, obj_fun, _, _ = fuzz.cluster.cmeans(data.T, c, m, error=e, maxiter=1000)
    else:
        # Usando a função implementada
        cntr, U, obj_fun, k = cmeans(data, c, m, e)

    new_data = np.zeros_like(data) # inicializar variável que irá receber os segmentos

    for j in range(data.shape[0]):
        i = np.argmax(U[:, j])
        new_data[j, :] = cntr[i, :]

    # Remodelar os dados segmentados para as dimensões da imagem
    new_img = (new_data * 255).astype(np.uint8).reshape(img.shape)

    # Plotar imagem segmentada
    plt.figure()
    plt.imshow(new_img)
    plt.title(f'Segmented Image with {c} Clusters')

# Plotar trajetória da função objetivo a cada iteração K

plt.figure()
plt.plot(obj_fun, ':x')
plt.xlabel('Number of iteration')
plt.ylabel('Value of the objective function')
plt.title('Objective Function Trajectories')
