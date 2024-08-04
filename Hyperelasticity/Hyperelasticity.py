import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['serif']

import numpy as np

DIM = 2
################################################################################
# Ops
################################################################################
def pow(x, p):
    return x**p

def exp(x):
    return np.exp(x)

def tensor_2by2_to4(a,b):

    c_array = []
    for i in range(DIM):
        for j in range(DIM):
            for p in range(DIM):
                for q in range(DIM):
                    c_array.append(a[i,j]*b[p,q])
    c = np.array(c_array).reshape((DIM*DIM, DIM*DIM))
    return c

################################################################################
# Mechanics
################################################################################

def material(youngs_mod, poisson_rat):
    mu  = youngs_mod/( (1+poisson_rat)*2 )
    lam = youngs_mod*poisson_rat/( (1+poisson_rat)*(1-2*poisson_rat) )
    kappa  = lam + 2/3 * mu
    return mu, lam, kappa

def FinvT(F):
    return np.linalg.inv(F).T
def dFinvT_dF(F):
    # dinvA_ij_dA_pq = invA_ip invA_qj
    # dinvA_ji_dA_pq = invA_pj invA_iq

    dF = FinvT(F)
    c_array = []
    for i in range(DIM):                # i =  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1
        for j in range(DIM):            # j =  0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1
            for p in range(DIM):        # p =  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1
                for q in range(DIM):    # q =  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
                    c_array.append(-dF[p,j]*dF[i,q]) # negative!
    c = np.array(c_array).reshape((DIM*DIM, DIM*DIM))
    return c

# verified
def J(F):
    return np.linalg.det(F)
def dJ_dF(F):
    return FinvT(F) * J(F)

def I1(F):
    return np.trace(F.T @ F)
def dI1_dF(F):
    return 2*F
def d2I1_dF2(F):
    #c_array = []
    #for i in range(DIM):                # i =  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1
    #    for j in range(DIM):            # j =  0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1
    #        for p in range(DIM):        # p =  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1
    #            for q in range(DIM):    # q =  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    #                if i == j and p == q and i == q:
    #                    c_array.append(1)
    #                else:
    #                    c_array.append(0)
    #d2I2 = np.array(c_array).reshape((DIM*DIM, DIM*DIM))
    return 2 * np.eye(DIM*DIM)
    #return d2I2

# verified
def I1_bar(F):
    return (pow(J(F),(-2./3.))) * I1(F)
def dI1_bar_dF(F):
    return -2/3 * pow(J(F),(-5/3)) * dJ_dF(F) * I1(F) \
            + pow(J(F),(-2/3)) * dI1_dF(F)
################################################################################
# FD
################################################################################
def FDcheck_2to2(func, params, input_idx):

    df_dx = np.zeros((DIM*DIM,DIM*DIM))
    f0 = (func(*params))

    FD_delta = 1.e-4

    # pertub input i,j
    idx = 0
    for i in range(DIM):
        for j in range(DIM):
            params[input_idx][i,j] += FD_delta
            fupp = (func(*params))
            params[input_idx][i,j] -= 2*FD_delta
            fdown = (func(*params))
            params[input_idx][i,j] += FD_delta

            df_dij = (fupp - fdown) / FD_delta / 2
            df_dx[:,idx] = df_dij.flatten()

            idx += 1
    return df_dx

def FDcheck_2to1(func, params, input_idx):
    df_dx = np.zeros((DIM,DIM))
    f0 = (func(*params))
    FD_delta = 1.e-4

    # pertub input i,j
    for i in range(DIM):
        for j in range(DIM):
            params[input_idx][i,j] += FD_delta
            fupp = (func(*params))
            params[input_idx][i,j] -= 2*FD_delta
            fdown = (func(*params))
            params[input_idx][i,j] += FD_delta

            df_dij = (fupp - fdown) / FD_delta/2
            df_dx[i,j] = df_dij

    return df_dx

################################################################################
# Plots
################################################################################
def plot_deformation(dW_dF_function, args):
    u11 = 0
    u21 = 0
    u12 = 0
    u22 = 0

    u11 = 0.1
    u21 = 0.2
    u12 = -0.3
    u22 = 0.2
    du_final = np.array([u11, u21, u12, u22]).reshape((DIM,DIM))
    du_final *= 5

    Nsteps = 10

    # input parameters
    youngs_mod = 1.0
    poisson_rat = 0.3

    # init
    mu, lam, kappa = material(youngs_mod, poisson_rat)
    x = []
    firstPK  = []
    secondPK = []
    cauchy   = []
    for i in range(Nsteps+1):
        # calculate increment of stress
        F = np.eye(DIM)
        F += du_final * (i/ Nsteps)
        P = dW_dF_function(F, *args)
        S = np.linalg.inv(F) @ P
        sigma = (1./J(F)) * P @ F.T

        # save plots
        x.append(F[0,0] - 1)
        firstPK.append(P[0,0])
        secondPK.append(S[0,0])
        cauchy.append(sigma[0,0])

    # plot
    fs = 12
    fig, ax = plt.subplots(layout='constrained', figsize = (6,4), dpi=200)

    ax.plot(x, firstPK, linestyle = "--", c="black", marker = '+', label="First-PK")
    ax.plot(x, secondPK, linestyle = "--", c="blue", marker = 'x', label="Second-PK")
    ax.plot(x, cauchy, linestyle = "--", c="red", marker = 'o', label="Cauchy")

    ax.legend(fontsize = fs,
              ncol=1,
              )
    ax.set_title("Hyperelastic Material: E = {}, v = {}".format(youngs_mod, poisson_rat))
    ax.set_ylabel('Stress Measure', fontsize = fs)
    ax.set_xlabel('Strain', fontsize = fs)
    plt.show()

