import numpy as np

import numpy as np
import scipy.optimize as optimize
from scipy.special import binom
import matplotlib.pyplot as plt
from time import time

def variance_upper_bound(m, n, K):
    """
        Evaluate the variance upper bound on variance of normalizer d(n,m), where K >= E[r(X)^2]. 
        This implementation is much less prone to numerical overflows than a brute computation
    """
    if K > 1:
        # Evaluate s_1 =  m^2 * K * [(n-m)!]^2/[(n-2m+1)! * n!]. Numerator has one additional term, which is calculated separately
        s_1 = (K-1)*m**2/(n-m+1) * np.prod([(n - m - l)/(n-l) for l in range(m-1)])

        # Evaluate s_1 * (1 + s_2/s_1 + s_3/s_2 * s_2/s_1 + ...x)
        out, tmp = 0, 1
        for l in range(1, m+1):
            out += s_1*tmp
            tmp *= (m - l)**2/((l+1)*(n-2*m + l + 1))*(K**(l+1) - 1)/(K**l - 1)
        return out
    else:
        return 0

def level_upper_bound(alpha_P, V, use_cantelli=True, taylor_approx=False):
    """
    Find minimizer of alpha_P/(1-delta) + V / delta**2 over delta in (0, 1)
    """
    if taylor_approx:
        return alpha_P*(1+V)
    if use_cantelli: 
        delta_bound = lambda delta: alpha_P / (1-delta) + V / (V + delta**2)
    else:
        delta_bound = lambda delta: alpha_P / (1-delta) + V / delta**2

    minimizer = optimize.minimize(delta_bound, x0 = 0.5, bounds=[(1e-12, 1-1e-12)])
    return delta_bound(minimizer['x'][0])

# Plot level guarantee for increasing m and various K
n_seq = [1000, 2000, 4000]
fig, axs = plt.subplots(1, len(n_seq), figsize=(10,3))
# plt.setp(axs, xlim=(0, max(n_seq)/10))
for i, n in enumerate(n_seq):
    for K in [1.0, 1.2, 1.5, 2, 3]:
        m_seq = [m for m in range(int(n/5))][::10]
        axs[i].plot(
            [m for m in m_seq if m < n], 
            [level_upper_bound(alpha_P=0.001, V=variance_upper_bound(m, n=n, K=K), use_cantelli=True) for m in m_seq if m < n],
            label=f"K={K}"
        )
        axs[i].set_ylim(0, 1)
        axs[i].set_title(f"n={n}")
    axs[-1].legend()
plt.savefig("revisions/bound_plot/level_bound.png")


# Plot level guarantee for K on x-axis
n_seq = [100, 1000, 10000]
alpha_seq = [0.05, 0.001, 0.0001]
fig, axs = plt.subplots(len(alpha_seq), len(n_seq), figsize=(10,10))
plt.rc('text', usetex=True)

for i, n in enumerate(n_seq):
    for row, alpha in enumerate(alpha_seq):
        for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            m = int(n**p)
            K_seq = np.linspace(1, 5, 30)
            axs[row, i].plot(
                K_seq, 
                [level_upper_bound(alpha_P=alpha, V=variance_upper_bound(m, n=n, K=K), use_cantelli=True) for K in K_seq],
                label=f"$m=n^{{{p}}}$"
            )
        axs[row, i].set_ylim(0, 1)
        axs[row, 0].set_ylabel("Level guarantee")
    axs[0, i].set_title(rf"$n={n}$")
    axs[-1, i].set_xlabel(rf"$K$")
axs[-1, -1].legend()
for ax, row in zip(axs[:,0], [f"$\\alpha_\\varphi = {a}$" for a in alpha_seq]):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

fig.tight_layout()
fig.subplots_adjust(left=0.15, top=0.95)

plt.savefig("revisions/bound_plot/fig_compare_K_alpha_and_m.png")

