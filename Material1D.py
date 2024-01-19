import numpy as np
from Material import Material

class Material1D(Material):
    def __init__(self, E):
        super().__init__() # create a material
        self.E = E

    def sigma_trial(self, eps_trial, eps_p):
        return self.E*(eps_trial - eps_p)

    def f_trial(self, sigma_tri, alpha):
        return np.abs(sigma_tri) - self.G(alpha)

    def C_elastic(self):
        return self.E

    def C_tangent(self, alpha):
        return self.E * self.dG(alpha) / (self.E + self.dG(alpha))

    def correct_trial_sigma(self, sigma_tri, delta_gamma):
        return sigma_tri - delta_gamma*self.E*np.sign(sigma_tri)

    def R(self, f_trial, alpha, delta_gamma):
        return f_trial - delta_gamma*self.E - self.G(alpha + delta_gamma) + self.G(alpha)

    def Jinv(self, f_trial, alpha, delta_gamma):
        J =  -self.E - self.dG(alpha + delta_gamma)
        return 1.0/J


class ExponentialHardening1D(Material1D):
    def __init__(self, E, sigma_Y, sigma_mu, delta):
        super().__init__(E) # create a 1D material
        self.sigma_Y = sigma_Y
        self.sigma_mu = sigma_mu
        self.delta = delta

    def G(self, alpha):
        return self.sigma_Y + (self.sigma_mu - self.sigma_Y)*(1 - np.exp(-self.delta*alpha))

    def dG(self, alpha):
        return (self.sigma_mu - self.sigma_Y) * self.delta * (np.exp(-self.delta*alpha))

class LinearHardening1D(Material1D):
    def __init__(self, E, sigma_Y, K):
        super().__init__(E) # create a 1D material
        self.sigma_Y = sigma_Y
        self.K = K

    def G(self, alpha):
        return self.sigma_Y + self.K*alpha

    def dG(self, alpha):
        return self.K

