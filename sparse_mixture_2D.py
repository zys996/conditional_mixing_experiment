import numpy as np
import scipy.stats as st
import utils
import matplotlib.pyplot as plt
from algorithm import ULA_optimizer

I_2 = np.identity(2)
m_1 = np.array([5, 5])
m_2 = np.array([-5, 5])
m_3 = np.array([-5, -5])
m_4 = np.array([5, -5])

def target_pdf(x):
    """
    calculate the target pdf given x
    """

    return 0.4 * st.multivariate_normal.pdf(x, m_1, I_2) + 0.4 * st.multivariate_normal.pdf(x, m_3, I_2) \
    + 0.1 * st.multivariate_normal.pdf(x, m_2, I_2) + 0.1 * st.multivariate_normal.pdf(x, m_4, I_2)

def target_cdf(x):
    """
    calculate the target pdf given x
    """

    return 0.4 * st.multivariate_normal.cdf(x, m_1, I_2) + 0.4 * st.multivariate_normal.cdf(x, m_3, I_2) \
    + 0.1 * st.multivariate_normal.cdf(x, m_2, I_2) + 0.1 * st.multivariate_normal.cdf(x, m_4, I_2)

def grad(x):
    """
    gradient oralce of log pdf (not pdf)
    """

    return np.array([(0.4 * (i - m_1) * st.multivariate_normal.pdf(i, m_1, I_2) + 0.4 * (i - m_3) * st.multivariate_normal.pdf(i, m_3, I_2) + \
    0.1 * (i - m_2) * st.multivariate_normal.pdf(i, m_2, I_2) + 0.1 * (i - m_4) * st.multivariate_normal.pdf(i, m_4, I_2)) / target_pdf(i) for i in x])

def sampling(n_round = 500):
    """
    sampling procedure
    """
    
    np.random.seed(0)
    optim = ULA_optimizer(eps = 1e-2, d = 2)
    
    # generate initial sampling points, for simplicity we will conduct this procedure in 4 quadrants respectively, which is helpful for KL div calculation
    x_1 = np.random.uniform(0, 10, (5000,1))
    x_2 = np.random.uniform(0, 10, (5000,1))
    x = np.concatenate((x_1, x_2), axis = 1)
    x_1 = np.random.uniform(-10, 0, (5000,1))
    x_2 = np.random.uniform(0, 10, (5000,1))
    x = np.concatenate((x, np.concatenate((x_1, x_2), axis = 1)))
    x_1 = np.random.uniform(-10, 0, (5000,1))
    x_2 = np.random.uniform(-10, 0, (5000,1))
    x = np.concatenate((x, np.concatenate((x_1, x_2), axis = 1)))
    x_1 = np.random.uniform(0, 10, (5000,1))
    x_2 = np.random.uniform(-10, 0, (5000,1))
    x = np.concatenate((x, np.concatenate((x_1, x_2), axis = 1)))

    # generate grid points
    grid = np.zeros((100,100,2))
    for i in range(100):
        grid[:,i,0] = np.linspace(-10, 10, 100)
        grid[i,:,1] = np.linspace(-10, 10, 100)
    x_grid = np.linspace(-10, 10, 100)
    y_grid = np.linspace(-10, 10, 100)

    kl_div_list_global = []
    kl_div_list_reg1 = []
    kl_div_list_reg2 = []
    kl_div_list_reg3 = []
    kl_div_list_reg4 = []

    for i in range(n_round):
        x = optim.update(x, grad(x))
        if(i % 10 == 0):
            print(i)
            kl_div_list_global.append(utils.calc_KL_divergence(x, grid, target_cdf, x_grid, y_grid, d = 2))
            kl_div_list_reg1.append(utils.calc_KL_divergence(x, grid[50:, 50:], target_cdf, x_grid[50:], y_grid[50:], is_condition=True, d = 2))
            kl_div_list_reg2.append(utils.calc_KL_divergence(x, grid[:50, 50:], target_cdf, x_grid[:50], y_grid[50:], is_condition=True, d = 2))
            kl_div_list_reg3.append(utils.calc_KL_divergence(x, grid[:50, :50], target_cdf, x_grid[:50], y_grid[:50], is_condition=True, d = 2))
            kl_div_list_reg4.append(utils.calc_KL_divergence(x, grid[50:, :50], target_cdf, x_grid[50:], y_grid[:50], is_condition=True, d = 2))

    path = "./sparse_mixture_2D_distribution.jpg"
    utils.draw_distribution_histogram(x, None, path, np.linspace(-20, 20, 8000), bw = 0.2, d = 2)

    plt.cla()
    plt.plot(range(0, n_round, 10), np.log(kl_div_list_global), label = "global")
    plt.plot(range(0, n_round, 10), np.log(kl_div_list_reg1), label = "conditioned on x > 0, y > 0")
    plt.plot(range(0, n_round, 10), np.log(kl_div_list_reg2), label = "conditioned on x < 0, y > 0")
    plt.plot(range(0, n_round, 10), np.log(kl_div_list_reg3), label = "conditioned on x < 0, y < 0")
    plt.plot(range(0, n_round, 10), np.log(kl_div_list_reg4), label = "conditioned on x > 0, y < 0")

    plt.tick_params(labelsize = 18)
    plt.xlabel("iteration", fontsize = 18)
    plt.ylabel(r"$\log(KL(p_t||p^{\star}))$", fontsize = 18)
    plt.legend(fontsize = 16, loc = 'upper right')
    plt.tight_layout()
    plt.savefig("./sparse_mixture_2D_div.jpg", dpi=600, bbox_inches='tight')

if(__name__ == "__main__"):
    sampling()
