import membership as mb
import numpy as np

def validation(especies, var_n, memb_n, tnorm, lim_meas, n, CF, classe):

    u_j = mb.membership_fn(especies, var_n, memb_n, tnorm, lim_meas, n)

    index = np.nanargmax(u_j*CF, 1)
    tipo = np.zeros(shape=len(index))

    for i, v in enumerate(index):
        tipo[i] = classe[v]

    return tipo