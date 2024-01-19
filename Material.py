from abc import ABC, abstractmethod
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['serif']

class Material(ABC):
    def __init__(self):
        self.strain_history = []
        self.stress_history = []

    @abstractmethod
    def G(self, alpha):
        """
        arguments:
            alpha: the current material state
        return:
            the value of the yield function
        """
        pass

    @abstractmethod
    def dG(self, alpha):
        """
        arguments:
            alpha: the current material state
        return:
            the gradient of the yield funciton wrt alpha
        """
        # return the yield function sensitivity dG_dself(self)
        pass

    @abstractmethod
    def R(self, f_trial, alpha, delta_gamma):
        """
        calculate the residual  of the current value of yield surface (f)
        and an approximate value due to the pururbation (alpha + delta_gamma)

        R = f_trial - sigma_trial(delta_gamma + eps_p) - G(alpha + delta_gamma) + G(alpha)

        When R -> 0,
            f_trial - sigma_trial(delta_gamma + eps_p) = G(alpha + delta_gamma) - G(alpha)
            ==>
            the delta_gamma increment produces a naive stress value that is approximately
            equal to its equivalent on the yield surface


        arguments:
            f_trial: current value of yield surface
            alpha: the current material state
            delta_gamma: current approximation of the increment
        return:
            residual (of dimension f_trial)
        """
        # return the difference between the trial state and the yield surface
        # used as a residual in newton rpahson
        pass

    @abstractmethod
    def Jinv(self, f_trial, alpha, delta_gamma):

        """
        Calculate the jacobian of the residual R wrt delta_gamma
        arguments:
            f_trial: current value of yield surface
            alpha: the current material state
            delta_gamma: current approximation of the increment
        return:
            the inverse jacobian of R wrt delta_gamma
        """
        pass

    @abstractmethod
    def sigma_trial(self, eps_trial, eps_p):
        """
        calculate stress state naively
        arguments:
            eps_trial: the trial strain
            eps_p: the current state of plastic stress
        return:
            the next step of stress, estimated naively (without regard to yield surface)
        """
        pass

    @abstractmethod
    def f_trial(self, sigma_tri, alpha):
        """
        evalluate the yield surface at a given state
        arguments:
            sigma_tri: the trial stress
            alpha: the current material state
        returns:
            value of yield surface
        """
        pass

    @abstractmethod
    def correct_trial_sigma(self, sigma_tri, delta_gamma):
        """
        shift sigma from your naive guess to a value on the yield surface
        arguments:
            sigma_tri: trial stress
            delta_gamma: the increment to shift stress
        returns:
            stress state on the yield suirface
        """
        pass

    @abstractmethod
    def C_elastic(self):
        """
        return:
            the elastic constitutive properties
        """
        pass

    @abstractmethod
    def C_tangent(self, alpha):
        """
        arguments:
            alpha: current material state
        return:
            par{sigma} / par{strain} at current state
        """
        pass

    def solve(self, eps_init, eps_final, eps_p, alpha, delta=0.1):
        """
        solve the material state
        arguments:
            eps_init: initial strain state
            eps_final: target/applied strain
            eps_p: initial state of plastic strain
            alpha: material internal variables
            delta: increment used in updating strain
        """
        eps_i = eps_init
        eps_increment = delta*(eps_final - eps_init)

        # log a history of strain/stress
        self.strain_history.append(eps_i)
        self.stress_history.append(self.sigma_trial(eps_i, eps_p))

        # loop until we reach the target strain
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
        """
        solve for the plastic flow increment given a yield
        arguments:
            f_trial: current state of yield funciton
            alpha: current material inernal state
        """
        max_iter = 10
        tol = 1.e-6
        i = 0
        delta_gamma = 0.0

        # begin newton raphson solve
        R = self.R(f_trial, alpha, delta_gamma)
        while(np.abs(R) > tol and i < max_iter):
            Jinv = self.Jinv(f_trial, alpha, delta_gamma)
            delta_gamma += -Jinv * R
            R = self.R(f_trial, alpha, delta_gamma)
            i += 1
        #print(f"converged in {i} iterations to {delta_gamma}")
        #print(f_trial / 1.1)
        return delta_gamma

    def plot_history(self, title, save_name=None):
        """
        plot strain-stress history for the entire material
        """
        fig, ax = plt.subplots(layout='constrained')
        ax.plot(self.strain_history, self.stress_history, 
                c="black", 
                linestyle="--", 
                marker="D")

        ax.set_ylabel(r'$\sigma$')
        ax.set_xlabel(r'$\varepsilon$')
        ax.set_title(title)
        if save_name is None:
            plt.show()
        else:
            plt.savefig(save_name)

