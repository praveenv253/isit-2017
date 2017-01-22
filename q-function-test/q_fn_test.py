#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import scipy.special as spl

def compute_bound(M):
    S = 1
    w = 0.1
    sigma2 = 1
    n = np.logspace(2, 5, 4)[:, np.newaxis]

    #M = 10
    delta = np.linspace(0, 0.25, 1000)

    # Define the space
    num_s = 10000
    s = np.linspace(0, S, num_s, endpoint=False)

    # Assume theta0 = 0, so that theta1 = 2*delta
    theta0 = 0  # But this is implicit, moving forward
    theta1 = 2 * delta[:, np.newaxis]

    # Define the continuous impulse response
    x0_c = np.zeros(s.size)
    i = np.abs(s) <= w/2
    x0_c[i] = w/2 - s[i]
    i = np.abs(S - s) <= w/2
    x0_c[i] = w/2 - (S - s[i])

    x1_c = np.zeros((delta.size, s.size))
    i = np.abs(s - theta1) <= w/2
    x1_c[i] = w/2 - np.abs(s - theta1)[i]
    i = np.abs(S - (s - theta1)) <= w/2
    x1_c[i] = w/2 - np.abs(S - (s - theta1)[i])

    # Compute mean sensor values by integrating the continuous impulse response
    x0 = np.zeros(M)
    x1 = np.zeros((delta.size, M))
    ds = S / num_s
    # j=0 is a special case
    i = (s >= (M - 0.5) * S / M) + (s < 0.5 * S / M)
    x0[0] = np.sum(x0_c[i]) * ds
    x1[:, 0] = np.sum(x1_c[:, i], axis=1) * ds
    for j in range(1, M):
        # Compute the expected value of the j'th sensor for all deltas
        i = (s >= (j - 0.5) * S / M) * (s < (j + 0.5) * S / M)
        x0[j] = np.sum(x0_c[i]) * ds
        x1[:, j] = np.sum(x1_c[:, i], axis=1) * ds

    #print(x0.shape)
    #print(x1.shape)
    #plt.plot(x1[100] - x0)
    #plt.show()

    dmin_by_2 = np.sqrt(np.sum((x1 - x0) ** 2, axis=-1)) / 2
    Pe = 0.5 - 0.5 * spl.erf(dmin_by_2 / np.sqrt(2 * sigma2 / M / n))

    bound = delta**2 * Pe
    return np.max(bound, axis=-1)

    #plt.plot(bound.T)
    #plt.show()

if __name__ == '__main__':
    Ms = np.arange(2, 50, 2)
    bound = []
    for M in Ms:
        print(M)
        bound.append(compute_bound(M))
    plt.plot(Ms, np.vstack(bound), linewidth=2)
    plt.title('Lower bound vs. number of sensors', fontsize=20)
    plt.xlabel('$m$', fontsize=20)
    plt.ylabel('$\mathfrak{M} = \delta^2 P_e$', fontsize=20)
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(('$\sigma_0^2/n = 10^{-2}$', '$\sigma_0^2/n = 10^{-3}$',
               '$\sigma_0^2/n = 10^{-4}$', '$\sigma_0^2/n = 10^{-5}$'),
              loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(('$\sigma_0^2/n = 10^{-2}$', '$\sigma_0^2/n = 10^{-3}$',
    #            '$\sigma_0^2/n = 10^{-4}$', '$\sigma_0^2/n = 10^{-5}$'),
    #           loc='upper right')
    plt.show()
