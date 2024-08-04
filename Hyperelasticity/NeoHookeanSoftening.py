import numpy as np
from Hyperelasticity import *
from NeoHookeanTest import W, dW_dF, d2W_dF2

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

def F_failure2D():
    u11 = 1.0
    u21 = 0.0
    u12 = 0.0
    u22 = 0.0

    du = np.array([u11, u21, u12, u22]).reshape((DIM,DIM))
    F = np.eye(DIM)
    F += du
    return F

def F_failure3D():
    u11 = 1.0
    u21 = 0.0
    u31 = 0.0
    u12 = 0.0
    u22 = 0.0
    u32 = 0.0
    u13 = 0.0
    u23 = 0.0
    u33 = 0.0
    du = np.array([u11, u21, u31, u12, u22, u32, u13, u23, u33]).reshape((DIM,DIM))

    F = np.eye(DIM)
    F += du
    return F

################################################################################
# Energy function with a critical limit
# doi.org/10.1016/j.jmps.2007.02.012
################################################################################
def Psi(F):
    return phi - phi * exp(-W(F) / phi)

def dPsi_dF(F):
    return dW_dF(F) * exp(-W(F) / phi)

def d2Psi_dF2(F):
    term1 =  d2W_dF2(F) * exp(-W(F) / phi)
    term2 =  tensor_2by2_to4(dW_dF(F),
                             (-1./phi) * dPsi_dF(F))
    #term2 = tensor_2by2_to4(dW_dF(F), dW_dF(F))
    #term2 *= -exp(-W(F) / phi) /phi
    #term2 = tensor_2by2_to4(dW_dF(F), dPsi_dF(F))
    #term2 *= -exp(-W(F) / phi) /phi
    return term1 + term2

################################################################################
# Main
################################################################################


# input parameters
youngs_mod = 1.0
poisson_rat = 0.3
mu, lam, kappa = material(youngs_mod, poisson_rat)
if(DIM == 2):
    F = F_init2D()
    F_at_failure = F_failure2D()
else :
    F = F_init3D()
    F_at_failure = F_failure3D()
phi = 10.0 * W(F)
#phi = W(F_at_failure)

def main():

    print(F)

    #print(F * F)
    #print(np.sum(E(F) * E(F)) )
    #print(W(F,mu,kappa))
    print("W, dW_dF, d2W_dF2")
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

    print("dPsi_dF")
    print(dPsi_dF(F) - FDcheck_2to1(Psi, (F,), 0))
    print()
    print("d2Psi_d2F")
    print(d2Psi_dF2(F, ))
    print(FDcheck_2to2(dPsi_dF, (F, ), 0))
    print(d2Psi_dF2(F, ) - FDcheck_2to2(dPsi_dF, (F, ), 0))
    print((d2Psi_dF2(F, ) - FDcheck_2to2(dPsi_dF, (F, ), 0))/d2Psi_dF2(F,))
    print()

    plot_deformation(dPsi_dF, ())

if __name__ == "__main__":
    main()
