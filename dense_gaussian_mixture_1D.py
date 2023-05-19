import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from algorithm import ULA_optimizer

def target_pdf(x):
    """
    calculate the target pdf given x
    """

    return 0.15 * st.norm.pdf(x, -5, 1) + 0.15 * st.norm.pdf(x, -2.5, 1) \
+ 0.3 * st.norm.pdf(x, 0, 1) + 0.2 * st.norm.pdf(x, 2.5, 1) + 0.2 * st.norm.pdf(x, 5, 1)

def target_cdf(x):
    """
    calculate the target pdf given x
    """

    return 0.15 * st.norm.cdf(x, -5, 1) + 0.15 * st.norm.cdf(x, -2.5, 1) \
+ 0.3 * st.norm.cdf(x, 0, 1) + 0.2 * st.norm.cdf(x, 2.5, 1) + 0.2 * st.norm.cdf(x, 5, 1)


def grad(x):
    """
    gradient oralce of log pdf (not pdf)
    """

    denom = 0.15 * np.exp(-(x + 5) ** 2 / 2.0) + 0.15 * np.exp(-(x + 2.5) ** 2 / 2.0) \
    + 0.3 * np.exp(-x ** 2 / 2.0) + 0.2 * np.exp(-(x - 2.5) ** 2 / 2.0) + 0.2 * np.exp(-(x - 5) ** 2 / 2.0)
    return (0.15 * (x + 5) * np.exp(-(x + 5) ** 2 / 2.0) + 0.15 * (x + 2.5) * np.exp(-(x + 2.5) ** 2 / 2.0) \
    + 0.3 * x * np.exp(-x ** 2 / 2.0) + 0.2 * (x - 2.5) * np.exp(-(x - 2.5) ** 2 / 2.0) + 0.2 * (x - 5) * np.exp(-(x - 5) ** 2 / 2.0)) / denom

def sampling(n_round = 5000):
    """
    sampling procedure
    """

    np.random.seed(0)
    optim = ULA_optimizer(eps = 1e-2, d = 1)        
    x = np.random.uniform(-10, 10, 20000)           #generate initial sampling points
    x_star = [target_pdf(i) for i in np.linspace(-10, 10, 500)]    #generate points on target pdf

    edge_list = np.array(np.linspace(-10, 10, 500))    # use these as grids to calculate KL divergence
    kl_div_list_global = []
    kl_div_list_reg1 = []
    kl_div_list_reg2 = []
    kl_div_list_reg3 = []
    kl_div_list_reg4 = []
    kl_div_list_reg5 = []
    interv = 100
    for i in range(n_round):

        x = optim.update(x, grad(x))
        if(i % interv == 0):
            print(i)
            kl_div_list_global.append(utils.calc_KL_divergence(x, edge_list, target_cdf))
            kl_div_list_reg1.append(utils.calc_KL_divergence(x, edge_list[:156], target_cdf, is_condition=True))
            kl_div_list_reg2.append(utils.calc_KL_divergence(x, edge_list[156:219], target_cdf, is_condition=True))
            kl_div_list_reg3.append(utils.calc_KL_divergence(x, edge_list[219:281], target_cdf, is_condition=True))
            kl_div_list_reg4.append(utils.calc_KL_divergence(x, edge_list[281:344], target_cdf, is_condition=True))
            kl_div_list_reg5.append(utils.calc_KL_divergence(x, edge_list[344:], target_cdf, is_condition=True))

    path = "./dense_mixture_1D_distribution.jpg"
    utils.draw_distribution_histogram(x, x_star, path, np.linspace(-10, 10, 500), bw = 0.4)

    plt.cla()
    plt.plot(range(0, n_round, interv), np.log(kl_div_list_global), label = "global")
    plt.plot(range(0, n_round, interv), np.log(kl_div_list_reg1), label = "conditioned on x < -3.75")
    plt.plot(range(0, n_round, interv), np.log(kl_div_list_reg2), label = "conditioned on -3.75 < x < -1.25")
    plt.plot(range(0, n_round, interv), np.log(kl_div_list_reg3), label = "conditioned on -1.25 < x < 1.25")
    plt.plot(range(0, n_round, interv), np.log(kl_div_list_reg4), label = "conditioned on 1.25 < x < 3.75")
    plt.plot(range(0, n_round, interv), np.log(kl_div_list_reg5), label = "conditioned on x > 3.75")
    plt.tick_params(labelsize = 18)
    plt.xlabel("iteration", fontsize = 18)
    plt.ylabel(r"$\log(KL(p_t||p^{\star}))$", fontsize = 18)
    plt.legend(fontsize = 16, loc = 'upper right')
    plt.tight_layout()
    plt.savefig("./dense_mixture_1D_div.jpg", dpi=600, bbox_inches='tight')

if(__name__ == "__main__"):
    sampling()
