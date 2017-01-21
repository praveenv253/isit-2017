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
    num_s = 1000
    s = np.linspace(0, S, M, endpoint=False)
    #s = np.linspace(0, S, num_s, endpoint=False)

    # Assume theta0 = 0, so that theta1 = 2*delta
    theta0 = 0  # But this is implicit, moving forward
    theta1 = 2 * delta[:, np.newaxis]

    # Define the sampled impulse response
    x0 = np.zeros(M)
    i = np.abs(s) <= w/2
    x0[i] = w/2 - s[i]
    i = np.abs(S - s) <= w/2
    x0[i] = w/2 - (S - s[i])

    x1 = np.zeros((delta.size, M))
    i = np.abs(s - theta1) <= w/2
    x1[i] = w/2 - np.abs(s - theta1)[i]
    i = np.abs(S - (s - theta1)) <= w/2
    x1[i] = w/2 - np.abs(S - (s - theta1)[i])

    #print(x0.shape)
    #print(x1.shape)
    #plt.plot(x1[100] - x0)
    #plt.show()

    dmin_by_2 = np.sqrt(np.sum((x1 - x0) ** 2, axis=-1)) / 2
    Pe = 0.5 - 0.5 * spl.erf(dmin_by_2 / np.sqrt(2 * M * sigma2 / n))

    bound = delta**2 * Pe
    return np.max(bound, axis=-1)

    #plt.plot(bound.T)
    #plt.show()

if __name__ == '__main__':
    Ms = np.arange(2, 50)
    bound = []
    for M in Ms:
        bound.append(compute_bound(M))
    plt.semilogy(np.vstack(bound))
    plt.title('Lower bound vs. number of sensors', fontsize=20)
    plt.xlabel('$m$', fontsize=20)
    plt.ylabel('$\mathcal{M} = \delta^2 P_e$', fontsize=20)
    plt.legend(('$\sigma_0^2/n = 10^{-2}$', '$\sigma_0^2/n = 10^{-3}$',
                '$\sigma_0^2/n = 10^{-4}$', '$\sigma_0^2/n = 10^{-5}$'),
               loc='best')
    plt.show()
