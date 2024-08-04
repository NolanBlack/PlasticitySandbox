from Hypelasticity import pow, material, FDcheck_2to2, FDcheck_2to1, tensor_2by2_to4, DIM, \
        I1, dI1_dF, d2I1_dF2, J, dJ_dF, FinvT, dFinvT_dF, I1_bar, dI1_bar_dF, \
        plot_deformation

# input parameters
youngs_mod = 1.0
poisson_rat = 0.3
mu, lam, kappa = material(youngs_mod, poisson_rat)

def F_init2D():
    u11 = 0.1
    u21 = 0.2
    u12 = -0.3
    u22 = 0.2
    u11 = 0
    u21 = 0
    u12 = 0
    u22 = 0

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
# Energy Density
################################################################################
def W(F) :
    W1 = mu/2 * (I1_bar(F) - DIM)
    W2 = kappa/4 * (pow(J(F) - 1,2))
    W3 = kappa/4 * ( pow(np.log(J(F)),2) )
    return W1 + W2 + W3

def dW_dF(F):
    return mu/2 * dI1_bar_dF(F) \
            + kappa/2 * (J(F) - 1) * dJ_dF(F) \
            + kappa/2 * np.log(J(F)) / J(F) * dJ_dF(F)
def dW_dF_ana(F):
    dW1_dF = mu * pow(J(F),(-2/3)) * ( F - (1/DIM)*I1(F) * FinvT(F) )
    dW2_dF = kappa/2 * J(F) * FinvT(F) * ( J(F) - 1)
    dW3_dF = kappa/2 * J(F) * FinvT(F) * ( 1/J(F) * np.log(J(F)) )
    return dW1_dF + dW2_dF + dW3_dF


def d2W_dF2(F):
    # W1
    d2I2 = d2I1_dF2(F)
    term1 = tensor_2by2_to4(mu * (-2/3)*pow(J(F),(-5/3))*dJ_dF(F) , 
                            F - (1/DIM)*I1(F) * FinvT(F) )
    term2 = mu * pow(J(F),(-2/3)) * ( 0.5*d2I2 
                                     - tensor_2by2_to4((1/DIM)*dI1_dF(F), FinvT(F))
                                     - (1/DIM)*I1(F)*dFinvT_dF(F) )
    d2W1_dF2 = term1 + term2

    # W2
    term1 = tensor_2by2_to4(kappa/2 * dJ_dF(F) * ( 2*J(F) - 1 ),
                                     FinvT(F))
    term2 = kappa/2 * J(F)*(J(F) - 1) * dFinvT_dF(F)
    d2W2_dF2 = term1 + term2

    # W3
    term1 = tensor_2by2_to4(kappa/2 * 1/J(F) * dJ_dF(F) ,
                            FinvT(F))
    term2 = kappa/2 * np.log(J(F)) * dFinvT_dF(F)
    d2W3_dF2 = term1 + term2

    return d2W1_dF2 + d2W2_dF2 + d2W3_dF2


################################################################################
# Main
################################################################################
def main():

    # input parameters
    youngs_mod = 1.0
    poisson_rat = 0.3

    # init
    mu, lam, kappa = material(youngs_mod, poisson_rat)
    if(DIM == 2):
        F = F_init2D()
    else :
        F = F_init3D()

    #print(dI1_bar_dF(F))
    #print(dI1_bar_dF(F) -  FDcheck_2to1(I1_bar, (F,), 0))
    #print(dFinvT_dF(F) - FDcheck_2to2(FinvT, (F,), 0))
    #print()
    #print()

    #print(FDcheck_2to1(W, (F), 0))
    #print(dW_dF(F) - dW_dF_ana(F))
    #print(dW_dF_ana(F) - FDcheck_2to1(W, (F), 0))
    #print()
    #print()

    #print(FDcheck_2to2(dI1_dF, (F, ), 0))
    #print(d2I1_dF2(F, ) -   FDcheck_2to2(dI1_dF, (F, ), 0))
    #print(FDcheck_2to2(dW_dF_ana, (F), 0))
    #print(d2W_dF2(F) -   FDcheck_2to2(dW_dF_ana, (F), 0))
    #print(F)
    #print(J(F))
    #print(I1(F))
    #print(I1_bar(F))
    #print(dJ_dF(F))
    #print(dI1_dF(F))
    #print(dI1_bar_dF(F))
    #print(dFinvT_dF(F))
    #print()
    #print()

    print(W(F))
    print(dW_dF_ana(F))
    print(d2W_dF2(F))

    err = (d2W_dF2(F) -   FDcheck_2to2(dW_dF_ana, (F), 0))
    print(np.amax(np.abs(err)))

    plot_deformation(dW_dF_ana, (mu, kappa))

if __name__ == "__main__":
    print("WORK IN PROGRESS")
    main()



#def d2W_dF2(F):
#    # W1
#    d2I2 = d2I1_dF2(F)
#    term1 = tensor_2by2_to4(mu * (-2/3)*pow(J(F),(-5/3))*dJ_dF(F) , 
#                            np.diag(np.diag(F)) - (1/3)*I1(F) * FinvT(F) )
#    term2 = mu * pow(J(F),(-2/3)) * ( d2I2 - tensor_2by2_to4((1/3)*dI1_dF(F), FinvT(F)) - (1/3)*I1(F)*dFinvT_dF(F) )
#    d2W1_dF2 = term1 + term2
#
#    # W2
#    term1 = tensor_2by2_to4(kappa/2 * dJ_dF(F) * ( J(F) - 1 ) + 
#                                     kappa/2 * J(F) * ( dJ_dF(F) ),
#                                     FinvT(F))
#    term2 = kappa/2 * J(F)*(J(F) - 1) * dFinvT_dF(F)
#    d2W2_dF2 = term1 + term2
#
#    # W3
#    term1 = tensor_2by2_to4(kappa/2 * dJ_dF(F) * 1/J(F) * np.log(J(F)) + 
#                                     kappa/2 * J(F) * pow(-J(F),-2) * dJ_dF(F) * np.log(J(F)) + 
#                                     kappa/2 * J(F) * 1/J(F) * 1/J(F) * dJ_dF(F) ,
#                                     FinvT(F))
#    term2 = kappa/2 * J(F)* 1/J(F) * np.log(J(F)) * dFinvT_dF(F)
#    d2W3_dF2 = term1 + term2
#
#    return d2W1_dF2 #+ d2W2_dF2 + d2W3_dF2













def foo():
    if DIM == 2:
        a_11, a_21, a_12, a_22 = a.flatten()
        b_11, b_21, b_12, b_22 = b.flatten()
        c_array =  [ a_11*b_11, a_11*b_21, a_11*b_12, a_11*b_22,
                     a_21*b_11, a_21*b_21, a_21*b_12, a_21*b_22,
                     a_12*b_11, a_12*b_21, a_12*b_12, a_12*b_22,
                     a_22*b_11, a_22*b_21, a_22*b_12, a_22*b_22 ]
    else :
        a_11, a_21, a_31, a_12, a_22, a_32, a_13, a_23, a_33 = a.flatten()
        b_11, b_21, b_31, b_12, b_22, b_32, b_13, b_23, b_33 = a.flatten()
        c_array = [a_11*b_11, a_11*b_21, a_11*b_31, a_11*b_12, a_11*b_22, a_11*b_32, a_11*b_13, a_11*b_23, a_11*b_33,
                   a_21*b_11, a_21*b_21, a_21*b_31, a_21*b_12, a_21*b_22, a_21*b_32, a_21*b_13, a_21*b_23, a_21*b_33,
                   a_31*b_11, a_31*b_21, a_31*b_31, a_31*b_12, a_31*b_22, a_31*b_32, a_31*b_13, a_31*b_23, a_31*b_33,
                   a_12*b_11, a_12*b_21, a_12*b_31, a_12*b_12, a_12*b_22, a_12*b_32, a_12*b_13, a_12*b_23, a_12*b_33,
                   a_22*b_11, a_22*b_21, a_22*b_31, a_22*b_12, a_22*b_22, a_22*b_32, a_22*b_13, a_22*b_23, a_22*b_33,
                   a_32*b_11, a_32*b_21, a_32*b_31, a_32*b_12, a_32*b_22, a_32*b_32, a_32*b_13, a_32*b_23, a_32*b_33,
                   a_13*b_11, a_13*b_21, a_13*b_31, a_13*b_12, a_13*b_22, a_13*b_32, a_13*b_13, a_13*b_23, a_13*b_33,
                   a_23*b_11, a_23*b_21, a_23*b_31, a_23*b_12, a_23*b_22, a_23*b_32, a_23*b_13, a_23*b_23, a_23*b_33,
                   a_33*b_11, a_33*b_21, a_33*b_31, a_33*b_12, a_33*b_22, a_33*b_32, a_33*b_13, a_33*b_23, a_33*b_33 ]

    # ij_pq --> F_ip F_qj
    # dF_1111  dF_1121  dF_1112  dF_1122 = dF_ip * dF_qj
    # dF_2111  dF_2121  dF_2112  dF_2122
    # dF_1211  dF_1221  dF_1212  dF_1222
    # dF_2211  dF_2221  dF_2212  dF_2222
    if DIM == 2:
        dF_11, dF_21, dF_12, dF_22 = FinvT(F).flatten()
        dF_tensor = [ dF_11*dF_11,  dF_11*dF_21,  dF_12*dF_11,  dF_12*dF_21,
                      dF_21*dF_11,  dF_21*dF_21,  dF_22*dF_11,  dF_22*dF_21,
                      dF_11*dF_12,  dF_11*dF_22,  dF_12*dF_12,  dF_12*dF_22,
                      dF_21*dF_12,  dF_21*dF_22,  dF_22*dF_12,  dF_22*dF_22]
    dF_tensor = -np.array(dF_tensor).reshape((DIM*DIM,DIM*DIM)) # negative!!
    #return dF_tensor
