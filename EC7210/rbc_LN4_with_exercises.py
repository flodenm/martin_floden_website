# -------------------------------------------------
# Code to find impulse-responses for core RBC model as presented
# in lecture notes 4
#
# Also solves some alternatives and exercises at the end of LN4
#
# Martin Flodén, Fall 2025
# -------------------------------------------------

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# -------------------------------------------------
# Choose version (mimic MATLAB input)
# -------------------------------------------------
prompt = (
    "Choose:\n"
    "1: Show IR to TFP shock as in Figure 1 in LN4\n"
    "2: Show IR to TFP shock when gamma = 0.5, as in Figure 2\n"
    "3: Show IR to TFP shock when rhoz = 0\n"
    "4: Show IR with k1 = 0.5*k_ss, as in Exercise 1\n"
    "5: Show IR to gvt spending shock as in Exercise 2\n"
    "6: Show IR to demand shock as in Exercise 3\n"
)
VERSION = int(input(prompt))

# We'll collect everything in a global-like dict, as in your earlier python code
GLOBAL = {}

# -------------------------------------------------
# Calibration
# -------------------------------------------------
alpha = 0.4          # capital share
delta = 0.02         # depreciation
if VERSION == 2:
    gamma = 0.5
else:
    gamma = 2.0      # utility parameter
ky_target = 12.0
beta = ky_target / (alpha + (1 - delta) * ky_target)
rhoz = 0.95
rhoG = 0.95
rhox = 0.95

# -------------------------------------------------
# Steady state
# -------------------------------------------------
ky_ss = alpha * beta / (1 - beta * (1 - delta))       # k/y
kh_ss = ky_ss ** (1 / (1 - alpha))                    # k/h
h_ss  = ((1 - alpha) / (1 - delta * ky_ss)) ** (1 / (1 + gamma))
k_ss  = kh_ss * h_ss
y_ss  = k_ss / ky_ss
c_ss  = y_ss - delta * k_ss
i_ss  = y_ss - c_ss
r_ss  = alpha * y_ss / k_ss - delta
w_ss  = (1 - alpha) * y_ss / h_ss

# -------------------------------------------------
# Shock paths and initial conditions
# -------------------------------------------------
T = 300
logz = np.zeros(T + 1)
G    = np.zeros(T + 1)
logx = np.zeros(T + 1)
k1   = k_ss  # default: start at steady-state capital

if VERSION in (1, 2):
    # persistent 1% TFP shock
    logz = (rhoz ** np.arange(T + 1)) * 0.01
elif VERSION == 3:
    # non-persistent TFP shock
    logz[0] = 0.01
elif VERSION == 4:
    # capital stock jumps down
    k1 = 0.5 * k_ss
elif VERSION == 5:
    # government consumption shock, 1% of steady-state output
    G = (rhoG ** np.arange(T + 1)) * 0.01 * y_ss
elif VERSION == 6:
    # demand shock in utility
    logx = (rhox ** np.arange(T + 1)) * 0.01
else:
    raise ValueError("Incorrect value for VERSION")

# -------------------------------------------------
# Initial guess (log of k(2..), c(1..), h(1..))
# -------------------------------------------------
xguess = np.log(np.concatenate([
    k_ss * np.ones(T),   # log k(2..T+1)
    c_ss * np.ones(T),   # log c(1..T)
    h_ss * np.ones(T)    # log h(1..T)
]))

# Fill GLOBAL (to imitate MATLAB globals)
GLOBAL.update(dict(
    alpha=alpha, beta=beta, gamma=gamma, delta=delta,
    k_ss=k_ss, h_ss=h_ss, c_ss=c_ss, y_ss=y_ss,
    logz=logz, G=G, logx=logx, k1=k1, T=T,
))

# -------------------------------------------------
# Helper: unpack x
# -------------------------------------------------
def unpack_x(x):
    """
    x[0:T]       = log k(2..T+1)
    x[T:2T]      = log c(1..T)
    x[2T:3T]     = log h(1..T)
    """
    G = GLOBAL
    T = G['T']
    alpha = G['alpha']
    logz  = G['logz']
    Gvt   = G['G']
    k1    = G['k1']
    c_ss  = G['c_ss']
    h_ss  = G['h_ss']

    # allocate
    k = np.empty(T + 1)
    c = np.empty(T + 1)
    h = np.empty(T + 1)

    # capital
    k[0] = k1
    k[1:] = np.exp(x[0:T])

    # consumption
    c[:-1] = np.exp(x[T:2*T])
    c[-1]  = c_ss

    # hours
    h[:-1] = np.exp(x[2*T:3*T])
    h[-1]  = h_ss

    # production
    y = np.exp(logz) * k**alpha * h**(1 - alpha)

    # investment (with government spending)
    i = y - c - Gvt

    # prices
    MPK = alpha * y / k
    MPL = (1 - alpha) * y / h
    r = MPK - GLOBAL['delta']
    w = MPL

    totalC = c + Gvt

    return k, c, h, y, i, r, w, totalC

# -------------------------------------------------
# Equilibrium conditions
# -------------------------------------------------
def evaluate_equilibrium_conditions(x):
    G = GLOBAL
    T = G['T']
    alpha = G['alpha']
    beta  = G['beta']
    gamma = G['gamma']
    delta = G['delta']
    logx  = G['logx']

    k, c, h, y, i, r, w, totalC = unpack_x(x)

    # demand shifter
    D = np.exp(logx)

    # 1) capital law of motion: k_{t+1} = (1-delta)k_t + y_t - c_t - G_t
    evalK = k[1:] - (1 - delta) * k[:-1] - i[:-1]

    # 2) Euler: (D_t c_{t+1}) / (D_{t+1} c_t) = beta (1 - delta + MPK_{t+1})
    MPK = alpha * y / k
    evalC = (D[:-1] * c[1:]) / (D[1:] * c[:-1]) - beta * (1 - delta + MPK[1:])

    # 3) Labor FOC: MPL_t * D_t / c_t = h_t^gamma
    evalH = (w * D / c)[:-1] - h[:-1]**gamma

    return np.concatenate([evalK, evalC, evalH])

# -------------------------------------------------
# Solve
# -------------------------------------------------
x_sol = fsolve(evaluate_equilibrium_conditions, xguess)

# -------------------------------------------------
# Reconstruct series
# -------------------------------------------------
k, c, h, y, i, r, w, totalC = unpack_x(x_sol)

# -------------------------------------------------
# Plot IRFs (3x3 as in MATLAB)
# -------------------------------------------------
irT = 80
scale = 100
t = np.arange(1, irT + 1)

fig, axes = plt.subplots(3, 3, figsize=(6, 4), dpi=150)

axes[0, 0].plot(t, scale * (y[:irT] / y_ss - 1))
axes[0, 0].set_title('output')
axes[0, 0].set_ylabel('% deviation from ss')

axes[0, 1].plot(t, scale * (c[:irT] / c_ss - 1))
axes[0, 1].set_title('consumption')

axes[0, 2].plot(t, scale * (h[:irT] / h_ss - 1))
axes[0, 2].set_title('hours')

# middle-left: shock panel
if VERSION <= 4:
    axes[1, 0].plot(t, scale * (np.exp(GLOBAL['logz'][:irT]) - 1))
    axes[1, 0].set_title('TFP')
    axes[1, 0].set_ylabel('% deviation from ss')
elif VERSION == 5:
    axes[1, 0].plot(t, 100 * GLOBAL['G'][:irT] / y[:irT])
    axes[1, 0].set_title('Gvt consumption')
    axes[1, 0].set_ylabel('% of output')
elif VERSION == 6:
    axes[1, 0].plot(t, scale * (np.exp(GLOBAL['logx'][:irT]) - 1))
    axes[1, 0].set_title('Demand')
    axes[1, 0].set_ylabel('% deviation from ss')

axes[1, 1].plot(t, scale * (i[:irT] / i_ss - 1))
axes[1, 1].set_title('investment')

axes[1, 2].plot(t, scale * (k[:irT] / k_ss - 1))
axes[1, 2].set_title('capital')

axes[2, 0].plot(t, scale * 4 * (r[:irT] - r_ss))
axes[2, 0].set_title('interest rate')
axes[2, 0].set_ylabel('ppt deviation from ss')

axes[2, 1].plot(t, scale * (w[:irT] / w_ss - 1))
axes[2, 1].set_title('wage')
axes[2, 1].set_ylabel('% deviation from ss')

axes[2, 2].plot(t, scale * (totalC[:irT] / c_ss - 1))
axes[2, 2].set_title('private + gvt consumption')

# zero lines and x-labels
for j in range(3):
    for i_ax in range(3):
        ax = axes[j, i_ax]
        ax.plot(t, np.zeros(irT), linewidth=1, color='black')
        if j == 2:
            ax.set_xlabel('quarters')

plt.tight_layout()
plt.show()
