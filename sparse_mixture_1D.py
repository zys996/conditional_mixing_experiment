import numpy as np
import scipy.stats as st
import utils
import matplotlib.pyplot as plt
from algorithm import ULA_optimizer

def target_pdf(x):
    """
    calculate the target pdf given x
    """

    return 0.1 * st.norm.pdf(x, 10, 1) + 0.9 * st.norm.pdf(x, -10, 1)

def target_cdf(x):
    """
    calculate the target pdf given x
    """

    return 0.1 * st.norm.cdf(x, 10, 1) + 0.9 * st.norm.cdf(x, -10, 1)

def grad(x):
    """
    gradient oralce of log pdf (not pdf)
    """

    denom = 0.1 * np.exp(-(x - 10) ** 2 / 2.0) + 0.9 * np.exp(-(x + 10) ** 2 / 2.0)
    return (0.1 * (x - 10) * np.exp(-(x - 10) ** 2 / 2.0) + 0.9 * (x + 10) * np.exp(-(x + 10) ** 2 / 2.0)) / denom

def sampling(n_round = 500):
    """
    sampling procedure
    """
    
    optim = ULA_optimizer(eps = 1e-2, d = 1)        
    x = np.random.uniform(-20, 20, 4000)           #generate initial sampling points
    x_star = [target_pdf(i) for i in np.linspace(-20, 20, 4000)]    #generate points on target pdf

    edge_list = np.array(np.linspace(-20, 20, 4000))    # use these as grids to calculate KL divergence
    kl_div_list_global = []
    kl_div_list_reg1 = []
    kl_div_list_reg2 = []

    for i in range(n_round):
        x = optim.update(x, grad(x))
        if(i % 10 == 0):
            print(i)
            kl_div_list_global.append(utils.calc_KL_divergence(x, edge_list, target_cdf))
            kl_div_list_reg1.append(utils.calc_KL_divergence(x, edge_list[:2000], target_cdf, is_condition=True))
            kl_div_list_reg2.append(utils.calc_KL_divergence(x, edge_list[2000:], target_cdf, is_condition=True))

    path = "./sparse_mixture_1D_distribution.jpg"
    utils.draw_distribution_histogram(x, x_star, path, np.linspace(-20, 20, 4000))

    plt.cla()
    plt.plot(range(0, n_round, 10), np.log(kl_div_list_global), label = "global")
    plt.plot(range(0, n_round, 10), np.log(kl_div_list_reg1), label = "conditioned on x < 0")
    plt.plot(range(0, n_round, 10), np.log(kl_div_list_reg2), label = "conditioned on x > 0")

    plt.tick_params(labelsize = 18)
    plt.xlabel("iteration", fontsize = 18)
    plt.ylabel(r"$\log(KL(p_t||p^{\star}))$", fontsize = 18)
    plt.legend(fontsize = 16)
    plt.tight_layout()
    plt.savefig("./sparse_mixture_1D_div.jpg", dpi=600, bbox_inches='tight')

if(__name__ == "__main__"):
    sampling()
