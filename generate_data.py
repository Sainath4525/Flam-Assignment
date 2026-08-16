"""
generate_data.py
-----------------
Generates xy_data.csv for the Flam SDE/R&D internship assignment.

Since the actual data file was not accessible, ground-truth values for
theta, M, X were chosen (within the specified bounds) and used to generate
sample points along the curve for 6 < t < 60. A small amount of Gaussian
noise is added to emulate real measured/sampled data.

GROUND TRUTH (for reference/testing only - normally unknown to the solver):
    theta_deg = 28.0   -> theta_rad = 0.488692...
    M         = 0.021
    X         = 63.4
"""

import numpy as np
import pandas as pd

# ---- Ground truth parameters (chosen within the assignment's bounds) ----
THETA_DEG = 28.0
THETA_RAD = np.deg2rad(THETA_DEG)
M_TRUE = 0.021
X_TRUE = 63.4

rng = np.random.default_rng(42)

# Sample t densely over the given range
t = np.linspace(6, 60, 300)

def x_of_t(t, theta, M, X):
    return t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X

def y_of_t(t, theta, M, X):
    return 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)

x = x_of_t(t, THETA_RAD, M_TRUE, X_TRUE)
y = y_of_t(t, THETA_RAD, M_TRUE, X_TRUE)

# Add small measurement noise so it behaves like real sampled data
noise_std = 0.15
x_noisy = x + rng.normal(0, noise_std, size=x.shape)
y_noisy = y + rng.normal(0, noise_std, size=y.shape)

df = pd.DataFrame({"x": x_noisy, "y": y_noisy})
df.to_csv("xy_data.csv", index=False)

print("Saved xy_data.csv with", len(df), "points")
print(f"Ground truth: theta_deg={THETA_DEG}, theta_rad={THETA_RAD:.6f}, M={M_TRUE}, X={X_TRUE}")
