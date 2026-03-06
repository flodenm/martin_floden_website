# =========================================================
# Code to solve Ramsey model numerically
#
# Assume: 
# Cobb Douglas production
# Log utility
# Constant productivity (g=0, A=1)
# Constant population (n=0, L=1)
# Initial capital stock = k0
# =========================================================


# =========================================================
# Suppose that beta suddenly falls (EX5 in LN2)
# =========================================================

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# =========================================================
# 1. Calibrate the model and find steady state
# one period = one year
# =========================================================
alpha = 0.4      # capital share
oldbeta = 0.96      # discount factor
beta = 0.90      # new discount factor
delta = 0.10     # depreciation rate

# steady state capital
kss = (alpha * beta / (1 - beta * (1 - delta))) ** (1 / (1 - alpha))
# steady state consumption
css = kss**alpha - delta * kss

oldkss = (alpha * oldbeta / (1 - oldbeta * (1 - delta))) ** (1 / (1 - alpha))
oldcss = oldkss**alpha - delta * oldkss

# We start in the old steady state
k0 = oldkss

# =========================================================
# 2. Find transition to new steady state
# =========================================================
T = 300  # number of periods in transition

# initial guess: logs of k(2..T+1) and c(1..T)
k_guess = np.log(kss) * np.ones(T)
c_guess = np.log(css) * np.ones(T)
x_guess = np.concatenate([k_guess, c_guess])

# Pass parameters to the evaluation function as a dictionary
params = {
    "alpha": alpha,
    "beta": beta,
    "delta": delta,
    "kss": kss,
    "css": css,
    "k0": k0,
    "T": T,
}


# =========================================================
# Function to evaluate equilibrium conditions
# =========================================================
def evaluate_equilibrium_conditions(x, params):
    """
    x[0:T]         = guess for log k(2), ..., log k(T+1)
    x[T:2*T]       = guess for log c(1), ..., log c(T)

    Returns a vector of length 2*T with:
    - capital accumulation equations
    - Euler equations
    """
    alpha = params["alpha"]
    beta = params["beta"]
    delta = params["delta"]
    kss = params["kss"]
    css = params["css"]
    k0 = params["k0"]
    T = params["T"]

    # unpack guesses
    k_log_next = x[0:T]          # k(2) ... k(T+1), in logs
    c_log = x[T:2*T]             # c(1) ... c(T), in logs

    # build full k and c vectors in levels
    k = np.empty(T + 1)
    k[0] = k0
    k[1:] = np.exp(k_log_next)

    # c(1..T) from guess, c(T+1) = css
    c = np.empty(T + 1)
    c[0:T] = np.exp(c_log)
    c[T] = css

    # production and MPK
    y = k**alpha
    MPK = alpha * y / k

    # capital accumulation for t = 1..T
    # k_{t+1} = (1-delta)k_t + y_t - c_t
    evalK = k[1:] - ((1 - delta) * k[:-1] + y[:-1] - c[:-1])

    # Euler equation for t = 1..T
    # c_{t+1} / c_t = beta * (1 - delta + MPK_{t+1})
    evalC = (c[1:] / c[:-1]) - beta * (1 - delta + MPK[1:])

    # stack them
    return np.concatenate([evalK, evalC])


# solve to find transition to steady state!
x, info, ier, msg = fsolve(
    evaluate_equilibrium_conditions, x_guess, args=(params,), full_output=True, xtol=1e-12
)

if ier != 1:
    print("WARNING: solver did not converge")
    print(msg)


# =========================================================
# 3. Recover k(t) and c(t)
# Suppose that we are in the old steady state in 10 periods
# before beta falls
# =========================================================
# k(1) = k0, k(2..T) = exp(x[0:T-1])
k = np.empty(T + 10)
k[0:10] = oldkss
k[10] = k0
k[11:] = np.exp(x[0 : T - 1])

# c(1..T) = exp(x[T:2*T])
c = np.empty(T + 10)
c[0:10] = oldcss
c[10:] = np.exp(x[T : 2 * T])

# =========================================================
# 4. Plot time paths
# =========================================================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(k, label="k_t")
plt.title("Capital")
plt.xlabel("Time")

plt.subplot(1, 2, 2)
plt.plot(c, label="c_t")
plt.title("Consumption")
plt.xlabel("Time")

plt.tight_layout()
plt.show()

# =========================================================
# 5. Phase diagram
# =========================================================
plt.figure()

# k' = k condition: c = f(k) - delta*k
kmax = delta ** (1 / (alpha - 1))   
k_grid = np.arange(0, kmax, kss / 100)  # simple grid
c_vals = k_grid**alpha - delta * k_grid

plt.plot(k_grid, c_vals, color="black", linewidth=2, label="\Delta k = 0")

# c' = c locus is just vertical line at kss
plt.plot([oldkss, oldkss], [0, 2 * oldcss], color="black", linewidth=2, label="old \Delta c = 0")
plt.plot([kss, kss], [0, 2 * css], color="red", linewidth=2, label="new \Delta c = 0")

plt.title("Phase diagram")
plt.xlabel("Capital stock, k")
plt.ylabel("Consumption, c")
plt.xlim(0, kmax)
plt.ylim(0, 2 * css)
plt.plot(k[:50], c[:50], color="red", marker=".")
plt.show()


# In[ ]:




