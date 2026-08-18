import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Load the NEW data
# ---------------------------------------------------------------------
df = pd.read_csv("xy_data.csv")
x_data = df["x"].to_numpy(dtype=float)
y_data = df["y"].to_numpy(dtype=float)

# IMPORTANT:
# The new CSV is not ordered by t. Therefore, rows must NOT be mapped
# directly to an evenly spaced t array.
#
# For the curve
#   x = t*cos(theta) - A(t)*sin(theta) + X
#   y = 42 + t*sin(theta) + A(t)*cos(theta)
# where A(t) = exp(M*|t|)*sin(0.3t),
# projection onto the rotated axes gives:
#   t = (x-X)*cos(theta) + (y-42)*sin(theta)
#   A = -(x-X)*sin(theta) + (y-42)*cos(theta)
#
# Hence the shuffled points can be fitted directly without knowing the
# original row-to-t correspondence.

def residuals(params):
    theta, M, X = params
    c, s = np.cos(theta), np.sin(theta)

    dx = x_data - X
    dy = y_data - 42.0

    t = dx*c + dy*s
    a = -dx*s + dy*c

    expected_a = np.exp(M*t) * np.sin(0.3*t)

    # Enforce the assignment condition 6 < t < 60.
    range_penalty = np.where(
        t < 6.0, 10.0*(6.0 - t),
        np.where(t > 60.0, 10.0*(t - 60.0), 0.0)
    )

    return np.concatenate([a - expected_a, range_penalty])


def model_xy(params, t):
    theta, M, X = params
    A = np.exp(M*np.abs(t)) * np.sin(0.3*t)

    x = t*np.cos(theta) - A*np.sin(theta) + X
    y = 42 + t*np.sin(theta) + A*np.cos(theta)
    return x, y


# ---------------------------------------------------------------------
# 2. Bounds
# ---------------------------------------------------------------------
lower_bounds = [np.deg2rad(0.001), -0.05, 0.001]
upper_bounds = [np.deg2rad(49.999), 0.05, 99.999]

# Mid-range initial guess.
initial_guess = [np.deg2rad(25.0), 0.0, 50.0]


# ---------------------------------------------------------------------
# 3. Fit
# ---------------------------------------------------------------------
result = least_squares(
    residuals,
    x0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    method="trf",
    xtol=1e-14,
    ftol=1e-14,
    gtol=1e-14,
    max_nfev=10000
)

theta_fit, M_fit, X_fit = result.x
theta_fit_deg = np.rad2deg(theta_fit)


# ---------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------
c, s = np.cos(theta_fit), np.sin(theta_fit)
dx = x_data - X_fit
dy = y_data - 42.0

t_implied = dx*c + dy*s
normal_obs = -dx*s + dy*c
normal_pred = np.exp(M_fit*t_implied) * np.sin(0.3*t_implied)

normal_error = normal_obs - normal_pred

print("=== Fit results ===")
print(f"theta = {theta_fit:.9f} rad  ({theta_fit_deg:.6f} deg)")
print(f"M     = {M_fit:.9f}")
print(f"X     = {X_fit:.9f}")
print(f"Cost  = {result.cost:.12e}")
print(f"RMSE (normal residual) = {np.sqrt(np.mean(normal_error**2)):.12e}")
print(f"Max |normal residual|  = {np.max(np.abs(normal_error)):.12e}")
print(f"Implied t range        = ({t_implied.min():.9f}, {t_implied.max():.9f})")


# ---------------------------------------------------------------------
# 5. Submission string
# ---------------------------------------------------------------------
latex_str = (
    r"\left(t\cos(%.6f)-e^{%.6f|t|}\sin(0.3t)\sin(%.6f)+%.6f,"
    r"42+t\sin(%.6f)+e^{%.6f|t|}\sin(0.3t)\cos(%.6f)\right)"
) % (
    theta_fit, M_fit, theta_fit, X_fit,
    theta_fit, M_fit, theta_fit
)

print("\n=== Desmos/LaTeX submission string ===")
print(latex_str)

with open("submission.txt", "w", encoding="utf-8") as f:
    f.write(f"theta (rad) = {theta_fit:.9f}\n")
    f.write(f"theta (deg) = {theta_fit_deg:.9f}\n")
    f.write(f"M = {M_fit:.9f}\n")
    f.write(f"X = {X_fit:.9f}\n")
    f.write(f"RMSE (normal residual) = {np.sqrt(np.mean(normal_error**2)):.12e}\n")
    f.write(f"Max |normal residual| = {np.max(np.abs(normal_error)):.12e}\n")
    f.write(f"Implied t range = ({t_implied.min():.9f}, {t_implied.max():.9f})\n\n")
    f.write("Desmos/LaTeX string:\n")
    f.write(latex_str + "\n")


# ---------------------------------------------------------------------
# 6. Plot fitted curve against shuffled data
# ---------------------------------------------------------------------
t_dense = np.linspace(6, 60, 1000)
x_fit, y_fit = model_xy(result.x, t_dense)

plt.figure(figsize=(8, 7))
plt.scatter(x_data, y_data, s=8, alpha=0.55, label="New CSV data (shuffled)")
plt.plot(x_fit, y_fit, linewidth=2.0, label="Fitted parametric curve")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Applied Materials / FLAM Parametric Curve Fit")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig("fit_plot.png", dpi=180)
plt.close()

print("\nSaved fit_plot.png")
