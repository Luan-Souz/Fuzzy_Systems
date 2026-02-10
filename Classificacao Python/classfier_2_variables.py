###################         Universidade do Estado do Amazonas          ###################
###################         Tópicos Especiais 1: Lógica Fuzy            ###################
###################         Classificador Fuzzy - 2 variáveis           ###################
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


#%% Separando  os atributos por especie
especies = []

especies.append(iris.loc[iris.tipo == 1])       # dados para setosa
especies.append(iris.loc[iris.tipo == 2])       # dados para versicolor
especies.append(iris.loc[iris.tipo == 3])       # dados para virginica

lim_meas = np.array([[4.3, 2],[7.9, 4.4]])          # limites para cada atributo

#%%Plotando 2 atributos

nomes = ['setosa', 'versicolor', 'virginica']

fig1 = plt.scatter(iris[iris.columns[0]],iris[iris.columns[1]],c=iris[iris.columns[4]])
plt.legend(handles= fig1.legend_elements()[0], labels = nomes)
plt.xlabel('$x_1$ (comprimento sepala)')
plt.ylabel('$x_2$ (largura sepala)')
plt.show()

#%% Imputando os dados


var_n = 2           # numero de variaveis
tnorm = 1           # 0: t-norma produtp; 1: t-norma minimo
p = 0.7             # procentagem para treinamento
N = 8               # numero de execucoes
memb_n = 8          # numero de funcoes de pertinencia
tm = int(50*p)

for iter_n in range(0,N):

    aux = np.random.permutation(50)
    n = list(itr.product(list(range(memb_n)), repeat=var_n))
    u_j = np.zeros(shape=(3, tm, len(n)))
    sum_u_j = np.zeros(shape=(3, len(n)))
    spe_rand = np.arange(float(3*var_n*tm)).reshape((3,tm,var_n))
    CF = np.zeros(shape=(len(n)))

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

    spe_rand_2part = list()

    for i in list(np.arange(lim_meas[0,0],lim_meas[1,0],0.01)):
        for j in list(np.arange(lim_meas[0,1],lim_meas[1,1],0.01)):
            spe_rand_2part.append([i, j])

    spe_rand_2part = np.array(spe_rand_2part)
    tipo = np.zeros(shape=(N, spe_rand_2part.shape[0]))
    tipo[iter_n] = validation(spe_rand_2part, var_n, memb_n, tnorm, lim_meas, n, CF, classe)

#%% Figuras

mapa_de_decisao = np.array(tipo[N-1]).reshape((361,241))
mapa_de_decisao = np.transpose(mapa_de_decisao)
plt.imshow(mapa_de_decisao, origin = 'lower',  extent = [4.3, 7.9, 2, 4.4], aspect= 1)
fig2 = plt.scatter(iris[iris.columns[0]],iris[iris.columns[1]],c=iris[iris.columns[4]],edgecolors= 'black')
plt.legend(handles= fig1.legend_elements()[0], labels = nomes)
plt.xlabel('$x_1$ (comprimento sepala)')
plt.ylabel('$x_2$ (largura sepala)')
plt.show()
