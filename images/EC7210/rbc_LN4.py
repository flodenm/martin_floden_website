# -------------------------------------------------
# Code to find impulse-responses for core RBC model as presented
# in lecture notes 4
#
# Martin Flodén, Fall 2025
# -------------------------------------------------

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# --------------------------
# Parameters & calibration
# --------------------------
alpha = 0.4
delta = 0.02
gamma = 2.0
ky_target = 12.0
beta = ky_target / (alpha + (1 - delta) * ky_target)
rho = 0.95
sigma = 0.01

# --------------------------
# Steady state
# --------------------------
ky_ss = alpha * beta / (1 - beta * (1 - delta))
kh_ss = ky_ss ** (1 / (1 - alpha))
h_ss  = ((1 - alpha) / (1 - delta * ky_ss)) ** (1 / (1 + gamma))
k_ss  = kh_ss * h_ss
y_ss  = k_ss / ky_ss
c_ss  = y_ss - delta * k_ss
i_ss  = y_ss - c_ss

# --------------------------
# Shock path and guesses
# --------------------------
T = 300
logz = 0.01 * (rho ** np.arange(T + 1))
k1 = k_ss                           # k(1) = k_ss
xguess = np.log(np.concatenate([
    k_ss * np.ones(T),              # log k(2..T+1)
    c_ss * np.ones(T),              # log c(1..T)
    h_ss * np.ones(T)               # log h(1..T)
]))

# Make some variables global to keep the code small
GLOBAL = dict(alpha=alpha, beta=beta, gamma=gamma, delta=delta,
              k_ss=k_ss, h_ss=h_ss, c_ss=c_ss, y_ss=y_ss,
              logz=logz, k1=k1, T=T)

def unpack_x(x):
    """Return k,c,h,y,i arrays given packed log-variables."""
    G = GLOBAL; T = G['T']
    k = np.empty(T + 1); c = np.empty(T + 1); h = np.empty(T + 1)

    k[0]  = G['k1']
    k[1:] = np.exp(x[0:T])              # k(2..T+1)
    c[:-1] = np.exp(x[T:2*T])           # c(1..T)
    c[-1]  = G['c_ss']                  # c(T+1) = ss
    h[:-1] = np.exp(x[2*T:3*T])         # h(1..T)
    h[-1]  = G['h_ss']                  # h(T+1) = ss

    y = np.exp(G['logz']) * k**G['alpha'] * h**(1 - G['alpha'])
    i = y - c
    return k, c, h, y, i

def evaluate_equilibrium_conditions(x):
    """Capital law, Euler, labor FOC (sizes T, T, T)."""
    G = GLOBAL; T = G['T']
    alpha, beta, gamma, delta = (
        G['alpha'], G['beta'], G['gamma'], G['delta'])

    k, c, h, y, i = unpack_x(x)
    r = alpha * y / k - delta
    w = (1 - alpha) * y / h

    evalK = k[1:] - (1 - delta) * k[:-1] - (y[:-1] - c[:-1]) 
    evalC = (c[1:] / c[:-1]) - beta * (1 + r[1:])            
    evalH = (w / c - h**gamma)[:-1]                          
    return np.concatenate([evalK, evalC, evalH])

# --------------------------
# Solve
# --------------------------
x = fsolve(evaluate_equilibrium_conditions, xguess)

# --------------------------
# Reconstruct series
# --------------------------
k, c, h, y, i = unpack_x(x)

# --------------------------
# Report IRF
# --------------------------
irT = 150
t = np.arange(1, irT + 1)

plt.figure(figsize=(12, 7))
plt.subplot(2, 3, 1); plt.plot(t, 100*(y[:irT]/y_ss-1)) 
plt.title('output'); plt.ylabel('% dev. from ss')
plt.subplot(2, 3, 2); plt.plot(t, 100*(c[:irT]/c_ss-1))
plt.title('consumption')
plt.subplot(2, 3, 3); plt.plot(t, 100*(h[:irT]/h_ss-1))
plt.title('hours')
plt.subplot(2, 3, 4); plt.plot(t, 100*(np.exp(logz[:irT])-1))
plt.title('TFP'); plt.ylabel('% dev. from ss')
plt.subplot(2, 3, 5); plt.plot(t, 100*(i[:irT]/i_ss-1))
plt.title('investment')
plt.subplot(2, 3, 6); plt.plot(t, 100*(k[:irT]/k_ss-1))
plt.title('capital')
plt.tight_layout()
plt.show()
