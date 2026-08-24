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

    mean_SHANG = np.zeros(T)
    mean_SHANGplus = np.zeros(T)
    mean_AGNES = np.zeros(T)
    mean_GD = np.zeros(T)
    mean_NAG = np.zeros(T)
    mean_SNAG = np.zeros(T)

    x_SHANG = np.ones(noofruns)
    v_SHANG = np.ones(noofruns)

    x_SHANGplus = np.ones(noofruns)
    v_SHANGplus = np.ones(noofruns)

    x_AGNES = np.ones(noofruns)
    v_AGNES = np.ones(noofruns)

    x_SNAG = np.ones(noofruns)
    z_SNAG = np.ones(noofruns)

    x_NAG = np.ones(noofruns)
    y_NAG = np.ones(noofruns)

    x_GD = np.ones(noofruns)

    val_SHANG = np.zeros(noofruns)
    val_SHANGplus = np.zeros(noofruns)
    val_AGNES = np.zeros(noofruns)
    val_GD = np.zeros(noofruns)
    val_NAG = np.zeros(noofruns)
    val_SNAG = np.zeros(noofruns)

    for n in range(T):
        if n % 50000 == 0:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("   Time is " + str(current_time) + ", steps = ", n)
        for i in range(effective_runs):

            # SHANG
            xcurrent_SHANG = x_SHANG[range(int(i * batch_size), int((i + 1) * batch_size))]
            v_prev_SHANG = v_SHANG[range(int(i * batch_size), int((i + 1) * batch_size))]
            val_SHANG[range(int(i * batch_size), int((i + 1) * batch_size))] = f(xcurrent_SHANG, deg)
            alpha_SHANG = 2 / (n + 1)
            gamma_SHANG = alpha_SHANG ** 2 * L * R ** 2
            beta_SHANG = (1 + sigma ** 2) * alpha_SHANG / gamma_SHANG
            if n == 0:
                alpha_prev_SHANG = alpha_SHANG
                gamma_prev_SHANG = gamma_SHANG
            else:
                alpha_prev_SHANG = 2 / n
                gamma_prev_SHANG = alpha_prev_SHANG ** 2 * L * R ** 2
            grad_SHANG = g(xcurrent_SHANG, sigma, deg)
            v_current_SHANG = v_prev_SHANG - (alpha_prev_SHANG / gamma_prev_SHANG) * grad_SHANG
            v_SHANG[range(int(i * batch_size), int((i + 1) * batch_size))] = v_current_SHANG
            x_SHANG[range(int(i * batch_size), int((i + 1) * batch_size))] = (1 / (1 + alpha_SHANG)) * xcurrent_SHANG + (
                        alpha_SHANG / (1 + alpha_SHANG)) * v_current_SHANG - beta_SHANG * (alpha_SHANG / (
                        1 + alpha_SHANG)) * grad_SHANG

            # SHANG++
            xcurrent_SHANGplus = x_SHANGplus[range(int(i * batch_size), int((i + 1) * batch_size))]
            v_prev_SHANGplus = v_SHANGplus[range(int(i * batch_size), int((i + 1) * batch_size))]
            val_SHANGplus[range(int(i * batch_size), int((i + 1) * batch_size))] = f(xcurrent_SHANGplus, deg)
            alpha_SHANGplus = 2 / (n + 1)
            rho_SHANGplus = 1.5
            modified_alpha_SHANGplus = alpha_SHANGplus / (1 + rho_SHANGplus * alpha_SHANGplus)
            gamma_SHANGplus = alpha_SHANGplus * modified_alpha_SHANGplus * L * R ** 2
            beta_SHANGplus = (1 + sigma ** 2) * alpha_SHANGplus / gamma_SHANGplus
            if n == 0:
                alpha_prev_SHANGplus = alpha_SHANGplus
                gamma_prev_SHANGplus = gamma_SHANGplus
            else:
                alpha_prev_SHANGplus = 2 / (n)
                modified_alpha_prev_SHANGplus = alpha_prev_SHANGplus / (1 + rho_SHANGplus * alpha_prev_SHANGplus)
                gamma_prev_SHANGplus = alpha_prev_SHANGplus * modified_alpha_prev_SHANGplus * L * R ** 2
            grad_SHANGplus = g(xcurrent_SHANGplus, sigma, deg)
            v_current_SHANGplus = v_prev_SHANGplus - (alpha_prev_SHANGplus / gamma_prev_SHANGplus) * grad_SHANGplus
            v_SHANGplus[range(int(i * batch_size), int((i + 1) * batch_size))] = v_current_SHANGplus
            x_SHANGplus[range(int(i * batch_size), int((i + 1) * batch_size))] = ((1 / (
                        1 + modified_alpha_SHANGplus)) * xcurrent_SHANGplus
                        + (modified_alpha_SHANGplus / (1 + modified_alpha_SHANGplus)) * v_current_SHANGplus
                        - beta_SHANGplus * (modified_alpha_SHANGplus / ( 1 + modified_alpha_SHANGplus)) * grad_SHANGplus)

            # AGNES
            # From https://github.com/kanangupta/AGNES
            xcurrent_AGNES = x_AGNES[range(int(i * batch_size), int((i + 1) * batch_size))]
            vcurrent_AGNES = v_AGNES[range(int(i * batch_size), int((i + 1) * batch_size))]
            val_AGNES[range(int(i * batch_size), int((i + 1) * batch_size))] = f(xcurrent_AGNES, deg)
            eta_AGNES = 1 / (L * (1 + 2 * sigma ** 2))
            gamma_AGNES = 1
            alpha_AGNES = eta_AGNES / R
            xprime_AGNES = xcurrent_AGNES + alpha_AGNES * vcurrent_AGNES
            rho_AGNES = (n) / (n + 5)
            grad_AGNES = g(xprime_AGNES, sigma, deg)
            v_AGNES[range(int(i * batch_size), int((i + 1) * batch_size))] = rho_AGNES * (
                        vcurrent_AGNES - gamma_AGNES * grad_AGNES)
            x_AGNES[range(int(i * batch_size), int((i + 1) * batch_size))] = xprime_AGNES - eta_AGNES * grad_AGNES

            # SNAG From: Algorthm2 in Paper: Julien Hermant, Marien Renaud, Jean-François Aujol, Charles Dossal,
            # and Aude Rondepierre. Gradient correlation is a key ingredient to accelerate SGD with momentum.
            #     (ICLR), 2025.
            xcurrent_SNAG = x_SNAG[range(int(i * batch_size), int((i + 1) * batch_size))]
            zcurrent_SNAG = z_SNAG[range(int(i * batch_size), int((i + 1) * batch_size))]
            val_SNAG[range(int(i * batch_size), int((i + 1) * batch_size))] = f(xcurrent_SNAG, deg)
            s_SNAG = 1 / (L * R)
            part_SNAG = (n ** 2) / (n + 1)
            eta_SNAG = (s_SNAG / R) * (n + 1) / 2
            beta_SNAG = 1
            alpha_SNAG = part_SNAG / (2 + part_SNAG)
            xprime_SNAG = alpha_SNAG * xcurrent_SNAG + (1 - alpha_SNAG) * zcurrent_SNAG
            grad_SNAG = g(xprime_SNAG, sigma, deg)
            x_SNAG[range(int(i * batch_size), int((i + 1) * batch_size))] = xprime_SNAG - s_SNAG * grad_SNAG
            z_SNAG[range(int(i * batch_size), int((i + 1) * batch_size))] = beta_SNAG * zcurrent_SNAG + (
                        1 - beta_SNAG) * xprime_SNAG - eta_SNAG * grad_SNAG

            # GD
            xcurrent_GD = x_GD[range(int(i * batch_size), int((i + 1) * batch_size))]
            val_GD[range(int(i * batch_size), int((i + 1) * batch_size))] = f(xcurrent_GD, deg)
            eta_GD = 1 / (L * R)
            grad_GD = g(xcurrent_GD, sigma, deg)
            x_GD[range(int(i * batch_size), int((i + 1) * batch_size))] = xcurrent_GD - eta_GD * grad_GD

            # NAG
            xcurrent_NAG = x_NAG[range(int(i * batch_size), int((i + 1) * batch_size))]
            ycurrent_NAG = y_NAG[range(int(i * batch_size), int((i + 1) * batch_size))]
            val_NAG[range(int(i * batch_size), int((i + 1) * batch_size))] = f(xcurrent_NAG, deg)
            rho_NAG = (n) / (n + 3)
            eta_NAG = 1 / (L * (1 + sigma ** 2))
            grad_NAG = g(ycurrent_NAG, sigma, deg)
            xNAG_prev = xcurrent_NAG
            x_NAG[range(int(i * batch_size), int((i + 1) * batch_size))] = ycurrent_NAG - eta_NAG * grad_NAG
            y_NAG[range(int(i * batch_size), int((i + 1) * batch_size))] = x_NAG[range(int(i * batch_size), int((
                                                                         i + 1) * batch_size))] + rho_NAG * (
                                                                         x_NAG[range(int(i * batch_size),
                                                                         int(( i + 1) * batch_size))] - xNAG_prev)

        mean_SHANG[n] = np.mean(val_SHANG)
        mean_SHANGplus[n] = np.mean(val_SHANGplus)
        mean_AGNES[n] = np.mean(val_AGNES)
        mean_NAG[n] = np.mean(val_NAG)
        mean_SNAG[n] = np.mean(val_SNAG)
        mean_GD[n] = np.mean(val_GD)
    return (mean_SHANG, mean_SHANGplus, mean_AGNES, mean_SNAG, mean_NAG, mean_GD)


globalnoofruns = 200  # for averaging over randomness
golablbatchsize = globalnoofruns

T = 100000  # number of steps the algorithm takes
sigmas = [0, 10, 50]
degs = [4, 16]

means_SHANG = np.zeros(shape=[len(degs), len(sigmas), T])
means_SHANGplus = np.zeros(shape=[len(degs), len(sigmas), T])
means_AGNES = np.zeros(shape=[len(degs), len(sigmas), T])
means_SNAG = np.zeros(shape=[len(degs), len(sigmas), T])
means_GD = np.zeros(shape=[len(degs), len(sigmas), T])
means_NAG = np.zeros(shape=[len(degs), len(sigmas), T])

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
        (means_SHANG[i, j, :],
         means_SHANGplus[i, j, :],
         means_AGNES[i, j, :],
         means_SNAG[i, j, :],
         means_NAG[i, j, :],
         means_GD[i, j, :]) = run_plot(deg, sigma, T=T,noofruns=noofruns, batch_size=batchsize)


for i in range(len(degs)):
  plt.figure()
  plt.title("d = "+str(degs[i]))
  plt.ylim(bottom = 1e-7, top = 2e2)
  gdsig0, = plt.loglog(means_GD[i,0, :], color = 'black')  #label = "GD, sigma = "+str(sigmas[0])
  gdsig1, = plt.loglog(means_GD[i,1, :],color = 'black', linestyle = '--') #label = "GD, sigma = "+str(sigmas[1])
  gdsig2, = plt.loglog(means_GD[i,2, :],color = 'black', linestyle = ':') #label = "GD, sigma = "+str(sigmas[2])

  nasig0, = plt.loglog(means_NAG[i,0, :], color = 'olive') #label = "NAG, sigma = "+str(sigmas[0])
  plt.loglog(means_NAG[i,1, :],color = 'olive', linestyle = '--', label = "NAG, sigma = "+str(sigmas[1]))
  plt.loglog(means_NAG[i,2, :],color = 'olive', linestyle = ':', label = "NAG, sigma = "+str(sigmas[2]))

  snasig0, = plt.loglog(means_SNAG[i, 0, :], color='orange')  # label = "SNAG, sigma = "+str(sigmas[0])
  plt.loglog(means_SNAG[i, 1, :], color='orange', linestyle='--', label="SNAG, sigma = " + str(sigmas[1]))
  plt.loglog(means_SNAG[i, 2, :], color='orange', linestyle=':', label="SNAG, sigma = " + str(sigmas[2]))

  agsig0, = plt.loglog(means_AGNES[i,0, :], color = 'blue') #label = "AGNES, sigma = "+str(sigmas[runs23])
  plt.loglog(means_AGNES[i,1, :], color = 'blue', linestyle = '--') #label = "AGNES, sigma = "+str(sigmas[1])
  plt.loglog(means_AGNES[i,2, :], color ='blue', linestyle = ':') #label = "AGNES, sigma = "+str(sigmas[2])

  hnagsig0, = plt.loglog(means_SHANG[i, 0, :], color='green')  # label = "SHANG, sigma = "+str(sigmas[0])
  plt.loglog(means_SHANG[i, 1, :], color='green', linestyle='--', label="SHANG, sigma = " + str(sigmas[1]))
  plt.loglog(means_SHANG[i, 2, :], color='green', linestyle=':', label="SHANG, sigma = " + str(sigmas[2]))

  ihnagsig0, = plt.loglog(means_SHANGplus[i, 0, :], color='red')  # label = "SHANG++, sigma = "+str(sigmas[0])
  plt.loglog(means_SHANGplus[i, 1, :], color='red', linestyle='--', label="SHANG++, sigma = " + str(sigmas[1]))
  plt.loglog(means_SHANGplus[i, 2, :], color='red', linestyle=':', label="SHANG++, sigma = " + str(sigmas[2]))

  plt.legend()
  sigma_labels = ["σ = "+str(sigma) for sigma in sigmas]
  legend1 = plt.legend([gdsig0, gdsig1, gdsig2], sigma_labels, loc=3)
  plt.legend([gdsig0, nasig0], ["SGD", "NAG"], loc=2)
  plt.legend([gdsig0, nasig0,snasig0, agsig0, hnagsig0, ihnagsig0], ["SGD", "NAG", "SNAG", "AGNES", "SHANG", "SHANG++"], loc=2)
  plt.gca().add_artist(legend1)
  plt.show()
