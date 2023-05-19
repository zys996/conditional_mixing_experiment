import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

def calc_KL_divergence(samples, grid, target_cdf, x_grid = None, y_grid = None, is_condition = False, d = 1):
    if d == 1:
        samples = [x for x in samples if grid[0] <= x <= grid[-1]]    # get the samples in the range
        f_t, b = np.histogram(samples, grid)  
        f_t = f_t / len(samples)        # get empirical frequency
        kl_div = 0
        denom = 1
        if is_condition:
            denom = target_cdf(grid[-1]) - target_cdf(grid[0])
        for i in range(len(grid) - 1):
            tmp = (target_cdf(grid[i + 1]) - target_cdf(grid[i])) / denom # calculate the (conditional) target cdf 
            # f_t[i] = 0 means p(x)log p(x)/q(x) = 0 here , tmp > 0 ensures calculation stability
            if(f_t[i] > 0 and tmp > 0):
                kl_div += f_t[i] * np.log(f_t[i] / tmp)
            # if target cdf is too small(tmp = 0 on double presicion) but we still get samples here, we take inf as 10^300 for approximation
            elif(f_t[i] > 0):
                kl_div += 300 * f_t[i]
    else:
        len_x, len_y, _  = grid.shape
        min_x = np.min(x_grid)
        max_x = np.max(x_grid)
        min_y = np.min(y_grid)
        max_y = np.max(y_grid)
        samples = np.array([sample for sample in samples if min_x <= sample[0] <= max_x and min_y <= sample[1] <= max_y])    # get the samples in the range
        f_t = np.histogram2d(x = samples[:, 0], y = samples[:, 1], bins = [x_grid, y_grid])
        f_t = f_t[0] / samples.shape[0]      # get empirical frequency
        kl_div = 0
        denom = 1
        if is_condition:
            denom = target_cdf([x_grid[-1], y_grid[-1]]) + target_cdf([x_grid[0], y_grid[0]]) - target_cdf([x_grid[-1], y_grid[0]]) - target_cdf([x_grid[0], y_grid[-1]])
        
        cdf = np.array([target_cdf(x) for x in grid.reshape((len_x*len_y, -1))]).reshape((len_x, len_y))
        tmp = (cdf[:-1,:-1] + cdf[1:, 1:] - cdf[:-1, 1:] - cdf[1:, :-1]) / denom
        tmp[tmp == 0] = 1e-100
        kl_div = np.sum(f_t * np.log((f_t + 1e-100) / tmp))
        # for i in range(len(x_grid) - 1):
        #     print(i)
        #     for j in range(len(y_grid) - 1):
        #         # calculate the (conditional) target cdf 
        #         tmp = (target_cdf([x_grid[i+1], y_grid[j+1]]) + target_cdf([x_grid[i], y_grid[j]]) - target_cdf([x_grid[i+1], y_grid[j]]) - target_cdf([x_grid[i], y_grid[j+1]])) / denom 
        #         # f_t[i] = 0 means p(x)log p(x)/q(x) = 0 here , tmp > 0 ensures calculation stability
        #         if(f_t[i][j] > 0 and tmp > 0):
        #             kl_div += f_t[i][j] * np.log(f_t[i][j] / tmp)
        #         # if target cdf is too small(tmp = 0 on double presicion) but we still get samples here, we take inf as 10^300 for approximation
        #         elif(f_t[i][j] > 0):
        #             kl_div += 300 * f_t[i][j]
    return kl_div

def draw_distribution_histogram(x_t, x_star, path, region, bw = 0.2, d = 1):
    if d == 1:
        sns.kdeplot(x_t, bw_adjust = bw)
        plt.plot(region, x_star)
        plt.legend(["sampling pdf", "target pdf"], fontsize = 18, loc = 'upper right')
        plt.xlabel("x", fontsize = 18)
        plt.ylabel("f(x)", fontsize = 18)
        plt.tick_params(labelsize = 18)
        plt.tight_layout()
        plt.savefig(path, dpi=600, bbox_inches='tight')
    else:
        sns.kdeplot(x = x_t[:,0], y = x_t[:, 1], bw_adjust = bw, fill = True)
        plt.xlabel("x", fontsize = 18)
        plt.ylabel("y", fontsize = 18)
        plt.tick_params(labelsize = 18)
        plt.xlim(-9,9)
        plt.ylim(-9,9)
        plt.tight_layout()
        plt.savefig(path, dpi=600, bbox_inches='tight')
