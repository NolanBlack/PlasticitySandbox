import numpy as np
from Hyperelasticity import pow, material, J, FDcheck_2to2, FDcheck_2to1, tensor_2by2_to4, DIM

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
# Strain energy wrt E
################################################################################

def E(F):
    return 0.5 * (F.T @ F - np.eye(DIM))

def dE_dF(F):
    c_array = []
    for i in range(DIM):                # i =  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1
        for j in range(DIM):            # j =  0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1
            for k in range(DIM):        # p =  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1
                for l in range(DIM):    # q =  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
                    dE_ijkl = 0.0
                    if(l == j):
                        dE_ijkl += F[k,i]
                    if(l == i):
                        dE_ijkl += F[k,j]
                    c_array.append(dE_ijkl)
    dE = np.array(c_array).reshape((DIM*DIM, DIM*DIM))
    dE *= 0.5
    return dE

def w(e) :
    lam = kappa - 2/3 * mu
    W1 = mu * np.sum(e * e)
    W2 =  0.5*lam*pow(np.trace(e), 2)
    return W1 + W2
def dw_de(e):
    lam = kappa - 2/3 * mu
    dw1_de = lam * np.trace(e) * np.eye(DIM)
    dw2_de = 2*mu*e
    return dw1_de + dw2_de
def d2w_de2(e):
    lam = kappa - 2/3 * mu
    #d2w1_de2 = lam * tensor_2by2_to4(np.eye(DIM),np.eye(DIM))
    C = np.zeros((DIM,DIM,DIM,DIM))
    c_array = []
    for i in range(DIM):                # i =  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1
        for j in range(DIM):            # j =  0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1
            for k in range(DIM):        # p =  0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1
                for l in range(DIM):    # q =  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
                    c_ijkl = 0.0
                    if (i == k) and (j ==l):
                        c_ijkl += 2*mu
                    if (i==j) and (k == l):
                        c_ijkl += lam
                    #c_array.append(d_ikjl)
                    C[i,j,k,l] = c_ijkl
    return C.flatten().reshape((DIM*DIM,DIM*DIM))
    #return np.array(c_array).reshape((DIM*DIM, DIM*DIM))
    #I = np.array(c_array).reshape((DIM*DIM, DIM*DIM))
    #d2w2_de2 = 2*mu * I
    #return d2w1_de2 + d2w2_de2

################################################################################
# Strain energy wrt F
################################################################################
def W(F) :
    lam = kappa - 2/3 * mu
    W1 = mu * np.sum(E(F) * E(F))
    W2 =  0.5*lam*pow(np.trace(E(F)), 2)
    return W1 + W2

def dW_dF(F):
    return tensor_2by4_to2(dw_de(E(F)), dE_dF(F))

# this is only true for small deformations!!
# D_{iJk L} = C_{IJKL} F_{iI} F_{kK} + δ_{ik} S_{JL}
def d2W_dF2_small_deformation(F):
    lam = kappa - 2/3 * mu
    C = d2w_de2(E(F)).flatten()
    S = dw_de(E(F))
    d = []
    for i in range(DIM):            
        for J in range(DIM):        
            for k in range(DIM):    
                for L in range(DIM):
                    D_iJkL = 0.0
                    if (i==k):
                        D_iJkL += S[J,L]
                    for I in range(DIM):
                        for K in range(DIM):
                            idx_IJKL = tensor4_idx(I,J,K,L)
                            D_iJkL += C[idx_IJKL] * F[i,I] * F[k,K]
                    d.append(D_iJkL)
    return np.array(d).reshape((DIM*DIM, DIM*DIM))

def S(F):
    e = E(F)
    D_ijkl = d2w_de2(e).flatten()
    return dw_de(e)

def d2W_dF2(F):
    e = E(F)
    D_ijkl = d2w_de2(e).flatten()
    s = dw_de(e)
    dP =  dP_dS(F, s) .flatten()
    dE = dE_dF(F).flatten()

    # PART 1
    # p_ijpq = dS_ij / dE_kl   :   dE_kl / dF_pq
    #        = dS_ij / dF_pq
    p_array = []
    for i in range(DIM):            
        for j in range(DIM):        
            for p in range(DIM):    
                for q in range(DIM):
                    p_ijpq = 0.0
                    # sum k,l
                    for k in range(DIM):            
                        for l in range(DIM):        
                            idx_ijkl = tensor4_idx(i,j,k,l)
                            idx_klpq = tensor4_idx(k,l,p,q)
                            p_ijpq += D_ijkl[idx_ijkl]*dE[idx_klpq]
                    p_array.append(p_ijpq)
    p_ijpq =  np.array(p_array)

    # PART 2
    # a_mnpq = dS_ij / dF_pq   :   dP_mn / dS_ij + S_nq del_mp
    #        = p_ijpq : dP_mn/dS_ij
    a_array = []
    for m in range(DIM):            
        for n in range(DIM):        
            for p in range(DIM):    
                for q in range(DIM):
                    a_mnpq = 0.0
                    if (m==p):
                        a_mnpq += s[n,q]
                    # sum i,j
                    for i in range(DIM):            
                        for j in range(DIM):        
                            idx_ijpq = tensor4_idx(i,j,p,q)
                            idx_mnij = tensor4_idx(m,n,i,j)
                            a_mnpq += p_ijpq[idx_ijpq]*dP[idx_mnij]
                    a_array.append(a_mnpq)
    return np.array(a_array).reshape((DIM*DIM, DIM*DIM))

    
################################################################################
# PK Stress
################################################################################

def P(F, S):
    P = 0.0 * S
    for i in range(DIM):            
        for I in range(DIM):        
            for J in range(DIM):    
                P[i,I] += F[i,J] * S[J,I]
    return P

def dP_dS(F, S):
    dP = []
    for i in range(DIM):            
        for I in range(DIM):        
            for K in range(DIM):    
                for P in range(DIM):
                    if(I == P) :
                        dP.append(F[i,K])
                    else:
                        dP.append(0.0)
    dP = np.array(dP).reshape((DIM*DIM, DIM*DIM))
    return dP

def tensor4_idx(i, j, p, q):
    return i*DIM*DIM*DIM + j*DIM*DIM + p*DIM + q


################################################################################
# Tensor Ops
################################################################################
def tensor_4by2_to2(A_ijkl, b_kl) :
    c = 0.0 * b_kl;
    A = A_ijkl.flatten()
    idx = 0
    for i in range(DIM):            
        for j in range(DIM):        
            for k in range(DIM):    
                for l in range(DIM):
                    c[i,j] += A[idx] * b_kl[k,l]
                    idx += 1
    return c

# b_ij A_ijkl = c_kl
def tensor_2by4_to2(b_ij, A_ijkl) :
    c = 0.0 * b_ij;
    A = A_ijkl.flatten()
    idx = 0
    for i in range(DIM):            
        for j in range(DIM):        
            for k in range(DIM):    
                for l in range(DIM):
                    c[k,l] += A[idx] * b_ij[i,j]
                    idx += 1
    return c


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

    # test nonzero inputs
    print("Inputs")
    e = E(F)
    s = dw_de(e)
    p = P(F, s)
    sigma = (1./J(F)) * F.T @ p.T
    print(F)
    print(E(F))
    print(p)
    print(s)
    print(sigma)
    print(np.linalg.inv(F) @ p - s)
    print(J(F) * sigma.T @ np.linalg.inv(F) - p)
    print()

    # test dE_dF
    print("dE_dF")
    print(dE_dF(F) -   FDcheck_2to2(E, (F, ), 0))
    print()

    # test dw_de
    #print(w(e,mu,kappa))
    #print(FDcheck_2to1(w, (e), 0))
    print("dw_de")
    print(dw_de(e) - FDcheck_2to1(w, (e,), 0))
    print("d2w_de2")
    print(d2w_de2(e) - FDcheck_2to2(dw_de, (e,), 0))
    print()

    print("dp_ds")
    #print(dP_dS(F, s))
    #print(FDcheck_2to2(P, (F, s), 1))
    print(dP_dS(F, s) - FDcheck_2to2(P, (F, s), 1))
    print()

    #print(F * F)
    #print(np.sum(E(F) * E(F)) )
    #print(W(F,mu,kappa))
    print("dW_dF")
    print(dW_dF(F) - FDcheck_2to1(W, (F,), 0))
    print()
    print("d2W_d2F")
    print(d2W_dF2(F) - FDcheck_2to2(dW_dF, (F,), 0))
    print()

    #e = E(F)
    #D_ijkl = d2w_de2(e).flatten()
    #s = dw_de(e)
    #dP =  dP_dS(F, s) .flatten()
    #dE = dE_dF(F).flatten()
    #print(d2w_de2(e))
    #print(dE_dF(F))
    #print(dP_dS(F, s))
    #print(d2W_dF2(F) / FDcheck_2to2(dW_dF, (F), 0))
    #print(FDcheck_2to2(dW_dF, (F), 0) / d2W_dF2(F) )

    gradu = F - np.eye(DIM)
    gradu /= 1000
    F0 = gradu + np.eye(DIM)
    dF = gradu

    P0 = dW_dF(F0)
    A = d2W_dF2(F0)
    P1 = (A @ dF.flatten()).reshape(DIM,DIM)
    print(P0)
    print(P1)
    print((P0 - P1)/P0)

if __name__ == "__main__":
    main()
