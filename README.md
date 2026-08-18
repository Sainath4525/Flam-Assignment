# FLAM SDE/R&D Internship — Parametric Curve Fitting (Updated)

## Problem

Recover the unknown constants `theta`, `M`, and `X` in:

```text
x(t) = t*cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta) + X
y(t) = 42 + t*sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta)
```

subject to:

- `0° < theta < 50°`
- `-0.05 < M < 0.05`
- `0 < X < 100`
- `6 < t < 60`

## Important change for the new CSV

The supplied CSV contains 1500 `(x, y)` points, but the rows are **shuffled** rather than ordered by increasing `t`.

Therefore, the old reference approach of using:

```python
t_data = np.linspace(6, 60, n)
```

is not valid for this new dataset.

Instead, the updated solution uses the geometry of the parametric model.

Define:

```text
A(t) = e^(M|t|) sin(0.3t)
```

Then:

```text
(x-X, y-42)
= t(cos(theta), sin(theta))
  + A(t)(-sin(theta), cos(theta))
```

Projecting an observed point onto the rotated axes gives:

```text
t = (x-X)cos(theta) + (y-42)sin(theta)

A = -(x-X)sin(theta) + (y-42)cos(theta)
```

The curve condition is therefore:

```text
A = e^(M|t|) sin(0.3t)
```

The three unknown parameters are fitted by bounded nonlinear least squares using this relation. This makes the solution independent of CSV row order.

## Result

The updated CSV gives:

```text
theta = 0.523598303 rad
theta = 29.999972932 deg
M     = 0.029999997
X     = 54.999998213
```

The recovered values are essentially:

```text
theta ≈ 30°
M     ≈ 0.03
X     ≈ 55
```

## Validation

```text
Number of data points: 1500
Implied t range: (6.049405473, 59.995170702)
Normal-residual RMSE: 3.486161151328e-06
Maximum absolute normal residual: 1.761505486408e-05
```

The very small residual confirms that the fitted parameters reproduce the supplied data to numerical precision.

## Files

- `xy_data.csv` — the supplied new CSV data.
- `fit_curve.py` — updated fitting solution that handles shuffled rows.
- `submission.txt` — fitted parameters and Desmos/LaTeX expression.
- `fit_plot.png` — visual comparison of the new data with the fitted curve.
- `README.md` — explanation of the updated method and results.

## How to run

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib
```

Then run:

```bash
python fit_curve.py
```

The script prints the fitted parameters and writes/updates:

- `submission.txt`
- `fit_plot.png`

## Desmos / LaTeX expression

```text
\left(t\cos(0.523598)-e^{0.030000|t|}\sin(0.3t)\sin(0.523598)+54.999998,42+t\sin(0.523598)+e^{0.030000|t|}\sin(0.3t)\cos(0.523598)\right)
```
