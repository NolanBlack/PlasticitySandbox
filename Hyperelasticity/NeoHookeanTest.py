import numpy as np
from Hyperelasticity import *

# input parameters
youngs_mod = 1.0
poisson_rat = 0.3
mu, lam, kappa = material(youngs_mod, poisson_rat)

################################################################################
# INIT
################################################################################
def F_init2D():
    u11 = 0.1
    u21 = 0.2
    u12 = -0.3
    u22 = 0.2
    #u11 = 0
    #u21 = 0
    #u12 = 0
    #u22 = 0

    du = np.array([u11, u21, u12, u22]).reshape((DIM,DIM))
    F = np.eye(DIM)
    F += du
    return F

def F_init3D():
    u11 = 0.1
    u21 = 0.2
    u31 = -0.1
    u12 = -0.5
    u22 = 0.4
    u32 = -0.1
    u13 = 0.0
    u23 = -0.2
    u33 = 0.3
    du = np.array([u11, u21, u31, u12, u22, u32, u13, u23, u33]).reshape((DIM,DIM))

    F = np.eye(DIM)
    F += du
    return F
################################################################################
# Strain energy wrt F
################################################################################
def W(F) :
    W1 = mu/2 * (I1(F) - DIM)
    W2 = -mu * np.log(J(F))
    W3 = lam/2 * ( pow(np.log(J(F)),2) )
    return W1 + W2 + W3

def dW_dF(F):
    dW1_dF = mu * F
    dW2_dF = -mu *FinvT(F)
    dW3_dF = lam * np.log(J(F)) * FinvT(F)
    return dW1_dF + dW2_dF + dW3_dF

def d2W_dF2(F):
    # W1
    d2W1_dF2 = mu * np.eye(DIM*DIM)

    # W2
    d2W2_dF2 = -mu * dFinvT_dF(F)

    # W3
    term1 = tensor_2by2_to4(lam/J(F) * dJ_dF(F) ,
                            FinvT(F))
    term2 = lam * np.log(J(F)) * dFinvT_dF(F)
    d2W3_dF2 = term1 + term2

    return d2W1_dF2 + d2W2_dF2 + d2W3_dF2
################################################################################
# Main
################################################################################

def main():

    if(DIM == 2):
        F = F_init2D()
    else :
        F = F_init3D()
    print(F)

    #print(F * F)
    #print(np.sum(E(F) * E(F)) )
    #print(W(F,mu,kappa))
    print(W(F))
    print(dW_dF(F))
    print(d2W_dF2(F))

    print("dW_dF")
    print(dW_dF(F) - FDcheck_2to1(W, (F,), 0))
    print()
    print("d2W_d2F")
    print(d2W_dF2(F, ))
    print(FDcheck_2to2(dW_dF, (F, ), 0))
    print(d2W_dF2(F, ) - FDcheck_2to2(dW_dF, (F, ), 0))
    print()

    #plot_deformation(dW_dF, ())


def check_hyperelasticity():
    if(DIM == 2):
        F = F_init2D()
    else :
        F = F_init3D()

    print("J")
    print(dJ_dF(F) - FDcheck_2to1(J, (F,), 0))
    print()

    print("I1")
    print(dI1_dF(F) - FDcheck_2to1(I1, (F,), 0))
    print()

    print("dI1_dF")
    print(d2I1_dF2(F) - FDcheck_2to2(dI1_dF, (F,), 0))
    print()

    print("I1bar")
    print(dI1_bar_dF(F) - FDcheck_2to1(I1_bar, (F,), 0))
    print()

    print("FinvT")
    print(dFinvT_dF(F) - FDcheck_2to2(FinvT, (F,), 0))
    print()

if __name__ == "__main__":
    main()
    #check_hyperelasticity()
