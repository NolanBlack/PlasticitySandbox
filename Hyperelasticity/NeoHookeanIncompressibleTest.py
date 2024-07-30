import numpy as np

DIM = 2

def pow(x, p):
    return x**p

def material(youngs_mod, poisson_rat):
    mu  = youngs_mod/( (1+poisson_rat)*2 )
    lam = youngs_mod*poisson_rat/( (1+poisson_rat)*(1-2*poisson_rat) )
    kappa  = lam + 2/3 * mu
    return mu, lam, kappa

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

def tensor_2by2_to4(a,b):

    c_array = []
    for i in range(DIM):
        for j in range(DIM):
            for p in range(DIM):
                for q in range(DIM):
                    c_array.append(a[i,j]*b[p,q])
    c = np.array(c_array).reshape((DIM*DIM, DIM*DIM))
    return c


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

# verified
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


def W(F, mu, kappa) :
    W1 = mu/2 * (I1_bar(F) - DIM)
    W2 = kappa/4 * (pow(J(F) - 1,2))
    W3 = kappa/4 * ( pow(np.log(J(F)),2) )
    return W1 + W2 + W3

def dW_dF(F, mu, kappa):
    return mu/2 * dI1_bar_dF(F) \
            + kappa/2 * (J(F) - 1) * dJ_dF(F) \
            + kappa/2 * np.log(J(F)) / J(F) * dJ_dF(F)
def dW_dF_ana(F, mu, kappa):
    dW1_dF = mu * pow(J(F),(-2/3)) * ( F - (1/DIM)*I1(F) * FinvT(F) )
    dW2_dF = kappa/2 * J(F) * FinvT(F) * ( J(F) - 1)
    dW3_dF = kappa/2 * J(F) * FinvT(F) * ( 1/J(F) * np.log(J(F)) )
    return dW1_dF + dW2_dF + dW3_dF


def d2W_dF2(F, mu, kappa):
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

def FDcheck_2to2(func, params, input_idx):

    df_dx = np.zeros((DIM*DIM,DIM*DIM))
    f0 = (func(*params))

    FD_delta = 1.e-3

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

    #print(FDcheck_2to1(W, (F, mu, kappa), 0))
    #print(dW_dF(F, mu, kappa) - dW_dF_ana(F, mu, kappa))
    #print(dW_dF_ana(F, mu, kappa) - FDcheck_2to1(W, (F, mu, kappa), 0))
    #print()
    #print()

    #print(FDcheck_2to2(dI1_dF, (F, ), 0))
    #print(d2I1_dF2(F, ) -   FDcheck_2to2(dI1_dF, (F, ), 0))
    #print(FDcheck_2to2(dW_dF_ana, (F, mu, kappa), 0))
    #print(d2W_dF2(F, mu, kappa) -   FDcheck_2to2(dW_dF_ana, (F, mu, kappa), 0))
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

    print(W(F, mu, kappa))
    print(dW_dF_ana(F, mu, kappa))
    print(d2W_dF2(F, mu, kappa))

    err = (d2W_dF2(F, mu, kappa) -   FDcheck_2to2(dW_dF_ana, (F, mu, kappa), 0))
    print(np.amax(np.abs(err)))


if __name__ == "__main__":
    main()



#def d2W_dF2(F, mu, kappa):
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
