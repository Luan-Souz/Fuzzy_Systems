import numpy as np
import skfuzzy as fuzzy

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
#%%
