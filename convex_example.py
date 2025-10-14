## Adapted from the AGNES codebase by Gupta et al. (MIT License).
## Source: https://github.com/kanangupta/AGNES

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

np.random.seed(52)

def f(x, deg):
    z = np.where(abs(x) <= 1, abs(x) ** deg, 1.0 + deg * (abs(x) - 1.0))
    return z

def df(x, deg):
    z = np.where(abs(x) <= 1, deg * abs(x) ** (deg - 2) * x, deg * x / abs(x))
    return z

def g(x, sigma, deg):
    return df(x, deg) * (1 + np.random.normal(0, sigma, x.size))

def run_plot(deg, sigma, T=1, noofruns=1, batch_size=1):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current state: sigma = " + str(sigma) + ", degree = " + str(deg) + ", time = " + str(current_time))
    L = deg * (deg - 1)
    R = (1 + sigma ** 2)
    effective_runs = int(noofruns / batch_size)

    mean_HNAG = np.zeros(T)
    mean_IHNAG = np.zeros(T)

    x_HNAG = np.ones(noofruns)
    v_HNAG = np.ones(noofruns)

    x_IHNAG = np.ones(noofruns)
    v_IHNAG = np.ones(noofruns)

    val_HNAG = np.zeros(noofruns)
    val_IHNAG = np.zeros(noofruns)

    for n in range(T):
        if n % 50000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("   Time is " + str(current_time) + ", steps = ", n)
        for i in range(effective_runs):
            # HNAG
            xcurrent_HNAG = x_HNAG[range(int(i * batch_size), int((i + 1) * batch_size))]
            v_prev_HNAG = v_HNAG[range(int(i * batch_size), int((i + 1) * batch_size))]
            val_HNAG[range(int(i * batch_size), int((i + 1) * batch_size))] = f(xcurrent_HNAG, deg)
            alpha_HNAG = 2 / (n + 1)
            gamma_HNAG = alpha_HNAG ** 2 * L * R ** 2
            beta_HNAG = R * alpha_HNAG / gamma_HNAG
            if n == 0:
                alpha_prev_HNAG = alpha_HNAG
                gamma_prev_HNAG = gamma_HNAG
            else:
                alpha_prev_HNAG = 2 / n
                gamma_prev_HNAG = alpha_prev_HNAG ** 2 * L * R ** 2
            grad_HNAG = g(xcurrent_HNAG, sigma, deg)
            v_current_HNAG = v_prev_HNAG - (alpha_prev_HNAG / gamma_prev_HNAG) * grad_HNAG
            v_HNAG[range(int(i * batch_size), int((i + 1) * batch_size))] = v_current_HNAG
            x_HNAG[range(int(i * batch_size), int((i + 1) * batch_size))] = (1 / (1 + alpha_HNAG)) * xcurrent_HNAG + (
                        alpha_HNAG / (1 + alpha_HNAG)) * v_current_HNAG - beta_HNAG * (alpha_HNAG / (
                        1 + alpha_HNAG)) * grad_HNAG

            # IHNAG
            xcurrent_IHNAG = x_IHNAG[range(int(i * batch_size), int((i + 1) * batch_size))]
            v_prev_IHNAG = v_IHNAG[range(int(i * batch_size), int((i + 1) * batch_size))]
            val_IHNAG[range(int(i * batch_size), int((i + 1) * batch_size))] = f(xcurrent_IHNAG, deg)
            alpha_IHNAG = 2 / (n + 1)
            rho_IHNAG = 1.5
            modified_alpha_IHNAG = alpha_IHNAG / (1 + rho_IHNAG * alpha_IHNAG)
            gamma_IHNAG = alpha_IHNAG * modified_alpha_IHNAG * L * R ** 2
            beta_IHNAG = R * alpha_IHNAG / gamma_IHNAG
            if n == 0:
                alpha_prev_IHNAG = alpha_IHNAG
                gamma_prev_IHNAG = gamma_IHNAG
            else:
                alpha_prev_IHNAG = 2 / (n)
                modified_alpha_prev_IHNAG = alpha_prev_IHNAG / (1 + rho_IHNAG * alpha_prev_IHNAG)
                gamma_prev_IHNAG = alpha_prev_IHNAG * modified_alpha_prev_IHNAG * L * R ** 2
            grad_IHNAG = g(xcurrent_IHNAG, sigma, deg)
            v_current_IHNAG = v_prev_IHNAG - (alpha_prev_IHNAG / gamma_prev_IHNAG) * grad_IHNAG
            v_IHNAG[range(int(i * batch_size), int((i + 1) * batch_size))] = v_current_IHNAG
            x_IHNAG[range(int(i * batch_size), int((i + 1) * batch_size))] \
                                         = ((1 / (1 + modified_alpha_IHNAG)) * xcurrent_IHNAG
                                            + (modified_alpha_IHNAG / (1 + modified_alpha_IHNAG)) * v_current_IHNAG
                                            - beta_IHNAG * (modified_alpha_IHNAG / (1 + modified_alpha_IHNAG)) * grad_IHNAG)

        mean_HNAG[n] = np.mean(val_HNAG)
        mean_IHNAG[n] = np.mean(val_IHNAG)

    return (mean_HNAG, mean_IHNAG)


globalnoofruns = 200  # for averaging over randomness
golablbatchsize = globalnoofruns

T = 100000  # number of steps the algorithm takes
sigmas = [0, 10, 50]
degs = [4, 16]

means_HNAG = np.zeros(shape=[len(degs), len(sigmas), T])
means_IHNAG = np.zeros(shape=[len(degs), len(sigmas), T])

for i in range(len(degs)):
    for j in range(len(sigmas)):
        deg = degs[i]
        sigma = sigmas[j]
        if sigma == 0:
            noofruns = 1
            batchsize = 1
        else:
            noofruns = globalnoofruns
            batchsize = golablbatchsize
        effectiveruns = int(noofruns / batchsize)
        means_HNAG[i, j, :], means_IHNAG[i, j, :] = run_plot(deg,sigma,T=T,noofruns=noofruns,batch_size=batchsize)


for i, d in enumerate(degs):
    for s, sigma in enumerate(sigmas):
        plt.figure()
        plt.title(f"d = {d}, σ = {sigma}")
        #     plt.ylim(bottom=1e-7, top=2e2)

        hnagsig, = plt.loglog(means_HNAG[i, s, :], color='green', linestyle='-')
        ihnagsig, = plt.loglog(means_IHNAG[i, s, :], color='red', linestyle='-')

        plt.legend([hnagsig, ihnagsig],
                   ["SHNAG", "ISHNAG"],
                   loc=1)

        path = f"./convex_example_deg{d}_sigma{sigma}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.show()
