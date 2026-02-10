###################         Universidade do Estado do Amazonas          ###################
###################         Tópicos Especiais 1: Lógica Fuzy            ###################
###################         Classificador Fuzzy - 4 variáveis           ###################
###################         Prof. Dr. Rodrigo Araújo                    ###################
###################         Aluno: Luan Souza                           ###################

#%% Funcoes: Membership e validação

import skfuzzy as fuzzy
import numpy as np

def membership_fn(especie, var_n, memb_n, tnorm, lim_meas, n):
    L = round(1/(memb_n - 1), ndigits=4)          # intervalo entre cada funcao de pertinencia
    x_norm = np.zeros(shape=(especie.shape[0],var_n))
    u = np.zeros(shape=(var_n,especie.shape[0],memb_n))
    m = len(especie[0])

    for i in range(0,var_n):
        x_norm[:,i] = (especie[:,i] - lim_meas[0][i]) / (lim_meas[1][i] - lim_meas[0][i])

        for j in range(memb_n):
            u[i][:,j] = fuzzy.trimf(x_norm[:,i], [round((j)*L-L,4), round((j)*L, 4), round((j)*L+L, 4)])

    u_j = np.ones((especie.shape[0], len(n)))

    if tnorm == 0:
        for j in range(len(n)):
            for i in range(0,var_n):
                u_j[:,j] = u_j[:,j] * u[i][:,n[j][i]]
    else:
        for j in range(0,len(n)):
            for i in range(0,var_n):
                u_j[:,j] = np.array([u_j[:,j], u[i][:,n[j][i]]]).min(axis=0)
    return u_j

def validation(especies, var_n, memb_n, tnorm, lim_meas, n, CF, classe):

    u_j = membership_fn(especies, var_n, memb_n, tnorm, lim_meas, n)

    index = np.nanargmax(u_j*CF, 1)
    tipo = np.zeros(shape=len(index))

    for i, v in enumerate(index):
        tipo[i] = classe[v]

    return tipo

#%% Importando bibliotecas necessárias

import pandas as pd
import matplotlib.pyplot as plt
import itertools as itr

np.seterr(all="ignore")

#%% Carregando os dados

iris = pd.read_table("iris.dat", sep=" ", header= None)
iris = iris.dropna(axis='columns')
iris.columns = ["comp sepala", "larg sepala", "comp petala", "larg petala", "tipo"]
 #attributes column [min - max]
 #1: sepal length  [4.3 - 7.9]
 #2: sepal width   [2.0 - 4.4]
 #3: petal length  [1.0 - 6.9]
 #4: petal width   [0.1 - 2.5]

#%% Separando  os atributos por especie
especies = []

especies.append(iris.loc[iris.tipo == 1])       # dados para setosa
especies.append(iris.loc[iris.tipo == 2])       # dados para versicolor
especies.append(iris.loc[iris.tipo == 3])       # dados para virginica

lim_meas = np.array([[4.3, 2, 1, 0.1], [7.9, 4.4, 6.9, 2.5]])          # limites para cada atributo


#%% Imputando os dados

var_n = 4           # numero de variaveis
tnorm = 0           # 0: t-norma produtp; 1: t-norma minimo
p = 0.7             # procentagem para treinamento
N = 25              # numero de execucoes
n_max = 8           # numero de funcoes de pertinencia
tm = int(50*p)

tipo_t = np.zeros(shape = (3, n_max, tm, N))        #
erro_t = np.zeros(shape = (3, n_max, n_max, N))

tipo_e = np.zeros(shape = (3, n_max, (50-tm), N))
erro_e = np.zeros(shape = (3, n_max, n_max, N))

S_t = np.zeros(shape=(3,n_max))
S_e = np.zeros(shape=(3,n_max))

err_t_mean = np.zeros(shape=(1, n_max))
err_e_mean = np.zeros(shape=(1, n_max))

#%%

for iter_n in range(0,N):
    aux = np.random.permutation(50)

    for memb_n in range(2,n_max):

        n = list(itr.product(list(range(memb_n)), repeat=var_n))
        CF = np.zeros(shape=(len(n)))
        u_j = np.zeros(shape=(3, tm, len(n)))
        sum_u_j = np.zeros(shape=(3, len(n)))
        spe_rand = np.zeros(shape=(3,tm,var_n))

        # Treinamento de Classes

        for k in range(0,3):

            spe_rand[k][::][::] = np.array(list(especies[k].values[aux[:tm], :var_n]))

            u_j[k] = membership_fn(spe_rand[k], var_n, memb_n, tnorm, lim_meas, n)
            sum_u_j[k,:] = sum(u_j[k])

        C = np.nanmax(sum_u_j, 0)
        classe = np.nanargmax(sum_u_j, 0)

        # Grau de certeza

        beta = np.sum(sum_u_j, 0)

        for i in range(len(n)):
             CF[i] = (sum_u_j[classe[i], i] - (beta[i] - sum_u_j[classe[i], i])/k) / beta[i]

        CF[np.isnan(CF)]=0

        # Validação utilizando dados de treinamento

        spe_rand_part_t = np.zeros(shape= (3,tm,var_n))

        for k in range(0,3):

            spe_rand_part_t[k][::][::] = np.array(list(especies[k].values[aux[:tm], :var_n]))
            tipo_t[k, memb_n][:,iter_n] = validation(spe_rand_part_t[k], var_n, memb_n, tnorm, lim_meas, n, CF, classe)
            S = 0
            for i in range(0,tm):
                if tipo_t[k, memb_n][i, iter_n] == k:
                    S+= 1

            erro_t[k, memb_n][:,iter_n] = (1 - S/(50-50*(1-p)))*100

        # Validação utilizando dados de Validação

        spe_rand_part_e = np.zeros(shape= (3,(50-tm),var_n))

        for k in range(0,3):

            spe_rand_part_e[k][::][::] = np.array(list(especies[k].values[aux[tm:], :var_n]))
            tipo_e[k, memb_n][:,iter_n] = validation(spe_rand_part_e[k], var_n, memb_n, tnorm, lim_meas, n, CF, classe)
            S = 0
            for i in range(0,50-tm):
                if tipo_e[k, memb_n][i, iter_n] == k:
                    S+= 1

            erro_e[k, memb_n][:,iter_n] = (1 - S/(50-tm))*100

        for k in range(0,3):
            S_t[k, memb_n] = np.mean(erro_t[k, memb_n])
            S_e[k, memb_n] = np.mean(erro_e[k, memb_n])

#%% Figuras

err_t_mean[:, :] = np.mean(S_t, 0)
err_e_mean[:, :] = np.mean(S_e, 0)


fig2, ax1 = plt.subplots(1,1)
ax1.plot(err_t_mean[0,0:], "--")
ax1.plot(err_e_mean[0,0:], "-")
ax1.grid(True)
plt.legend(['Modelo Treinamento', 'Modelo avaliação'])
plt.ylabel('Taxa de erro (%)')
plt.xlabel('Número de funções de pertinência')
plt.xlim(2,n_max-1)
plt.style.use('seaborn-v0_8-whitegrid')
plt.show()



#%%
