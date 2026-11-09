# -------------------------------------------------
# EC7210 Macroeconomics
#
# Simulate the core RBC model from lecture notes 4.
# Python counterpart to rbc_with_simulation.m.
#
# Martin Floden, 2026
# Stockholm University
# -------------------------------------------------

import numpy as np
from scipy.optimize import fsolve
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
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
sigma = 0.009  # standard deviation of quarterly TFP innovations

# --------------------------
# Steady state
# --------------------------
ky_ss = alpha * beta / (1 - beta * (1 - delta))
kh_ss = ky_ss ** (1 / (1 - alpha))
h_ss = ((1 - alpha) / (1 - delta * ky_ss)) ** (1 / (1 + gamma))
k_ss = kh_ss * h_ss
y_ss = k_ss / ky_ss
c_ss = y_ss - delta * k_ss

# --------------------------
# Small shock path and guesses
# --------------------------
T = 300
logz = (sigma / 100) * rho ** np.arange(T + 1)
k1 = k_ss
xguess = np.log(np.concatenate([
    k_ss * np.ones(T),  # log k(2..T+1)
    c_ss * np.ones(T),  # log c(1..T)
    h_ss * np.ones(T),  # log h(1..T)
]))


def unpack_x(x):
    """Return k, c, h, and y, including the terminal period."""
    k = np.concatenate(([k1], np.exp(x[:T])))
    c = np.concatenate((np.exp(x[T:2*T]), [c_ss]))
    h = np.concatenate((np.exp(x[2*T:]), [h_ss]))
    y = np.exp(logz) * k**alpha * h**(1 - alpha)
    return k, c, h, y


def evaluate_equilibrium_conditions(x):
    """Capital accumulation, Euler equation, and labor supply."""
    k, c, h, y = unpack_x(x)
    r = alpha * y / k - delta
    w = (1 - alpha) * y / h

    evalK = k[1:] - (1 - delta) * k[:-1] - (y[:-1] - c[:-1])
    evalC = c[1:] / c[:-1] - beta * (1 + r[1:])
    evalH = (w / c - h**gamma)[:-1]
    return np.concatenate([evalK, evalC, evalH])


# --------------------------
# Solve and fit policy rules
# --------------------------
x, solver_info, exit_flag, solver_message = fsolve(
    evaluate_equilibrium_conditions, xguess, xtol=1e-12, full_output=True
)
max_residual = np.max(np.abs(solver_info['fvec']))
if exit_flag != 1 or max_residual > 1e-10:
    raise RuntimeError(f'Equilibrium solve failed: {solver_message} '
                       f'(maximum residual {max_residual:.2e})')

k, c, h, y = unpack_x(x)

# Regress log k(t+1) and log h(t) on a constant, log(k(t)/k_ss), and log z(t).
X = np.column_stack([np.ones(T), np.log(k[:-1] / k_ss), logz[:-1]])
b = np.linalg.lstsq(X, np.log(k[1:]), rcond=None)[0]
bh = np.linalg.lstsq(X, np.log(h[:-1]), rcond=None)[0]

# --------------------------
# Simulate the economy
# --------------------------
burnT = 1000
simT = 100000
seed = None  # Set to an integer for reproducible results in Python.
rng = np.random.default_rng(seed)

simk = np.zeros(burnT + simT + 1)
simz = np.zeros(burnT + simT + 1)
simeps = rng.normal(0, sigma, burnT + simT)
simk[0] = np.log(k_ss)
for t in range(burnT + simT):
    simz[t + 1] = rho * simz[t] + simeps[t]
    simk[t + 1] = b[0] + b[1] * (simk[t] - simk[0]) + b[2] * simz[t + 1]

# Discard the burn-in. Capital has one extra observation for k(t+1).
simK = np.exp(simk[burnT:])
simTFP = np.exp(simz[burnT + 1:])
simH = np.exp(bh[0] + bh[1] * np.log(simK[:-1] / k_ss) + bh[2] * np.log(simTFP))
simY = simTFP * simK[:-1]**alpha * simH**(1 - alpha)
simC = simY + (1 - delta) * simK[:-1] - simK[1:]
simI = simY - simC

# --------------------------
# HP-filter the log series and report moments
# --------------------------
log_series = 100 * np.log(np.column_stack([simY, simC, simI, simH]))
hp_lambda = 1600
second_difference = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(simT - 2, simT))
hp_system = eye(simT, format='csc') + hp_lambda * (second_difference.T @ second_difference)
trend = spsolve(hp_system.tocsc(), log_series)
cycle = log_series - trend

std = cycle.std(axis=0, ddof=1)
relative_std = std / std[0]
correlations = np.corrcoef(cycle, rowvar=False)
series_names = ('Output', 'Consumption', 'Investment', 'Hours')

print(f'\nHP-filtered model moments (lambda = {hp_lambda})')
print(f"{'Variable':<12} {'Volatility rel. to output':>25} {'Correlation with output':>24}")
for name, volatility, correlation in zip(series_names, relative_std, correlations[:, 0]):
    print(f'{name:<12} {volatility:>25.2f} {correlation:>24.2f}')

# --------------------------
# Plot a short simulation sample
# --------------------------
sample = slice(999, 1050)  # MATLAB quarters 1000:1050 after burn-in
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (name, series) in zip(axes.flat, [
    ('Output', simY), ('Consumption', simC), ('Hours', simH),
    ('TFP', simTFP), ('Investment', simI), ('Capital', simK),
]):
    ax.plot(series[sample])
    ax.set_title(name)
fig.tight_layout()
plt.show()
