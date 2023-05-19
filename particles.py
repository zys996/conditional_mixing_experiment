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

def sampling(n_tot = 20000, n_particle = 1, n_round = 1000):
    """
    sampling procedure
    """
    
    np.random.seed(0)
    optim = ULA_optimizer(eps = 1e-2, d = 2)
    
    # generate initial sampling points, for simplicity we will conduct this procedure in 4 quadrants respectively, which is helpful for KL div calculation
    x_1 = np.random.uniform(-10, 10, (n_particle,1))
    x_2 = np.random.uniform(-10, 10, (n_particle,1))
    x = np.concatenate((x_1, x_2), axis = 1)

    for i in range(n_round):
        x = optim.update(x, grad(x))
        if(i % 50 == 0):
            print(i)
    
    samples = x.copy()
    n_sample = n_tot // n_particle

    for i in range(n_sample - 1):
        x = optim.update(x, grad(x))
        samples = np.concatenate((samples, x))
        if(i % 1000 == 0):
            print(i)

    path = "./particles_"+str(n_particle)+".jpg"
    plt.cla()
    utils.draw_distribution_histogram(samples, None, path, np.linspace(-20, 20, 8000), bw = 0.2, d = 2)

if(__name__ == "__main__"):
    # sampling(n_particle = 1)
    # sampling(n_particle = 10)
    # sampling(n_particle = 2000)

    samples = []
    for i in range(100000):
        tmp = np.random.uniform()
        if tmp < 0.4:
            samples.append(np.random.multivariate_normal(m_1, I_2))
        elif 0.4 <= tmp < 0.8:
            samples.append(np.random.multivariate_normal(m_3, I_2))
        elif 0.8 <= tmp < 0.9:
            samples.append(np.random.multivariate_normal(m_2, I_2))
        else:
            samples.append(np.random.multivariate_normal(m_4, I_2))

    path = "./target.jpg"
    plt.cla()
    utils.draw_distribution_histogram(np.array(samples), None, path, np.linspace(-20, 20, 8000), bw = 0.2, d = 2)

