from Material import Material
from Material1D import Material1D, ExponentialHardening1D, LinearHardening1D

def linear_hardening_1d():
    E = 1.0 # modulus of elasticity
    sigma_Y = 1.0 # the yield stress
    K = 0.1 # post-yield modulus
    material = LinearHardening1D(E, sigma_Y, K)
    return material

def exponential_hardening_1d():
    E = 1.0 # modulus of elasticity
    sigma_Y = 1.0 # the yield stress
    sigma_mu = 4.0 # the stress at infinity
    delta = 0.2 # rate of exponential increase
    material = ExponentialHardening1D(E, sigma_Y, sigma_mu, delta)
    return material


def main():

    #material = linear_hardening_1d()
    material = exponential_hardening_1d()

    eps_init = 0.0 # initial strain
    eps_p = 0.0 # state of plastic strain at initial
    alpha = 0.0 # internal strain state
    solver_delta = 0.1 # delta for incrementing strain

    # initial load step
    eps_final = 10.0 # target strain
    eps_i, eps_p, alpha = material.solve(eps_init, eps_final, eps_p, alpha, solver_delta)

    # unload to initial config
    eps_final = 0.0
    eps_i, eps_p, alpha = material.solve(eps_i, eps_final, eps_p, alpha, solver_delta)

    # load again
    eps_final = 20.0
    eps_i, eps_p, alpha = material.solve(eps_i, eps_final, eps_p, alpha, solver_delta)

    # plot
    material.plot_history("demo_1D", "results/demo1D.png")

if __name__ == "__main__":
    main()
