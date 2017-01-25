#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import scipy.special as spl

def compute_bound(M):
    S = 1
    w = 0.1
    sigma2 = 1
    n = np.logspace(4, 7, 4)[:, np.newaxis]

    #M = 10
    delta = np.linspace(0, 0.25, 1000, endpoint=False)

    # Define the space
    num_s = 50000
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
    Pe = 0.5 - 0.5 * spl.erf(dmin_by_2 / np.sqrt(sigma2 / M / n))

    bound = delta**2 * Pe
    plt.figure()
    plt.semilogy(bound.T)
    plt.savefig('opt-%d.png' % M)
    plt.close()

    return np.max(bound, axis=-1)


if __name__ == '__main__':
    Ms = np.r_[2:60:2]
    bound = []
    for M in Ms:
        print(M)
        bound.append(compute_bound(M))
    plt.semilogy(Ms, np.vstack(bound), linewidth=2)
    plt.title('Numerically computed lower bound vs.\nnumber of sensors',
              fontsize=20)
    plt.xlabel('Number of sensors, $m$', fontsize=20)
    plt.ylabel('Lower bound, $\sup_{0<\delta<S/4} \delta^2 P_e$', fontsize=20)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.text(40, 4e-3, '$n = 1$', fontsize=18)
    plt.text(40, 3e-4, '$n = 10$', fontsize=18)
    plt.text(40, 2e-6, '$n = 100$', fontsize=18)
    plt.text(40, 2e-7, '$n = 1000$', fontsize=18)
    plt.tight_layout()
    plt.show()
