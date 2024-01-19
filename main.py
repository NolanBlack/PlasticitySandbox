from abc import ABC, abstractmethod
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['serif']

"""
    alpha: internal state of material
    delta_gamma: 
"""

class Material(ABC):
    def __init__(self):
        self.strain_history = []
        self.stress_history = []

    @abstractmethod
    def G(self, alpha):
        # return the yield function G(self)
        pass

    @abstractmethod
    def dG(self, alpha):
        # return the yield function sensitivity dG_dself(self)
        pass

    @abstractmethod
    def R(self, f_trial, alpha, delta_gamma):
        # return the difference between the trial state and the yield surface
        pass

    @abstractmethod
    def Jinv(self, f_trial, alpha, delta_gamma):
        # return the inverse jacobian of the R
        pass

    @abstractmethod
    def sigma_trial(self, eps_trial, eps_p):
        pass

    @abstractmethod
    def f_trial(self, sigma_tri, alpha):
        pass

    @abstractmethod
    def correct_trial_sigma(self, sigma_tri, delta_gamma):
        pass

    @abstractmethod
    def C_elastic(self):
        pass

    @abstractmethod
    def C_tangent(self, alpha):
        pass

    def solve(self, eps_init, eps_final, eps_p, alpha, delta=0.1):
        eps_i = eps_init
        eps_increment = delta*(eps_final - eps_init)

        self.strain_history.append(eps_i)
        self.stress_history.append(self.sigma_trial(eps_i, eps_p))

        while(np.sign(eps_increment) * (eps_i - eps_final) < 0 ):
            # increment strain
            eps_i += eps_increment

            # evaluate stress/yield state (naive)
            sigma_tri = self.sigma_trial(eps_i, eps_p) # E*(eps - eps_p)
            f_tri = self.f_trial(sigma_tri, alpha) # yield function

            # check yield surface
            if(f_tri <= 0.0): # elastic
                sigma_i = sigma_tri
                C = self.C_elastic()
            else: # plastic update
                # solve for the 
                delta_gamma = self.NRstep(f_tri, alpha)
                sigma_i = self.correct_trial_sigma(sigma_tri, delta_gamma) # update stress
                eps_p += delta_gamma * np.sign(sigma_tri) # update plastic strain
                alpha += delta_gamma # internal state update
                C = self.C_tangent(alpha)

            #print(eps_i, sigma_tri, " --> ", eps_i, sigma_i)

            self.strain_history.append(eps_i)
            self.stress_history.append(sigma_i)

        return eps_i, eps_p, alpha

    def NRstep(self, f_trial, alpha):
        max_iter = 10
        tol = 1.e-6

        i = 0
        delta_gamma = 0.0
        R = self.R(f_trial, alpha, delta_gamma)
        while(np.abs(R) > tol and i < max_iter):
            Jinv = self.Jinv(f_trial, alpha, delta_gamma)
            delta_gamma += -Jinv * R
            R = self.R(f_trial, alpha, delta_gamma)
            i += 1
        #print(f"converged in {i} iterations to {delta_gamma}")
        #print(f_trial / 1.1)
        return delta_gamma

    def plot_history(self):
        fig, ax = plt.subplots(layout='constrained')
        ax.plot(self.strain_history, self.stress_history, 
                c="black", 
                linestyle="--", 
                marker="D", 
                label="C1111")

        #ax.legend(fontsize = 12,
        #          ncol=1,
        #          )

        ax.set_ylabel(r'$\sigma$')
        ax.set_xlabel(r'$\varepsilon$')
        #ax.set_xticks(np.arange(0, 1.1, 0.1))
        plt.show()
        pass

class Material_1D(Material):
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

class ExponentialHardening_1D(Material_1D):
    def __init__(self, E, sigma_Y, sigma_mu, delta):
        super().__init__(E) # create a 1D material
        self.sigma_Y = sigma_Y
        self.sigma_mu = sigma_mu
        self.delta = delta

    def G(self, alpha):
        return self.sigma_Y + (self.sigma_mu - self.sigma_Y)*(1 - np.exp(-self.delta*alpha))

    def dG(self, alpha):
        return (self.sigma_mu - self.sigma_Y) * self.delta * (np.exp(-self.delta*alpha))

class LinearHardening_1D(Material_1D):
    def __init__(self, E, sigma_Y, K):
        super().__init__(E) # create a 1D material
        self.sigma_Y = sigma_Y
        self.K = K

    def G(self, alpha):
        return self.sigma_Y + self.K*alpha

    def dG(self, alpha):
        return self.K


def main():


    # material properties
    E = 1.0 # modulus of elasticity
    sigma_Y = 1.0 # the yield stress
    sigma_mu = 4.0 # the stress at infinity
    delta = 0.2 # rate of exponential increase
    material = ExponentialHardening_1D(E, sigma_Y, sigma_mu, delta)

    # material properties
    E = 1.0 # modulus of elasticity
    sigma_Y = 1.0 # the yield stress
    K = 0.1 # post-yield modulus
    #material = LinearHardening_1D(E, sigma_Y, K)


    eps_init = 0.0
    eps_p = 0.0
    alpha = 0.0
    eps_final = 10.0
    solver_delta = 0.1
    eps_i, eps_p, alpha = material.solve(eps_init, eps_final, eps_p, alpha, solver_delta)
    eps_final = 0.0
    eps_i, eps_p, alpha = material.solve(eps_i, eps_final, eps_p, alpha, solver_delta)
    eps_final = 20.0
    eps_i, eps_p, alpha = material.solve(eps_i, eps_final, eps_p, alpha, solver_delta)
    material.plot_history()

if __name__ == "__main__":
    main()
