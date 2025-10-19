# NewtonLab — Truncated & Modified Newton (CLI)

**Fast Newton-type solvers for unconstrained optimization** featuring:
- **Truncated Newton (TN):** Hessian-free optimization using CG on Hessian–vector products
- **Modified Newton (MN):** Full Hessian with positive-definite correction, solved via CG or Cholesky

Fully configurable through JSON with support for exact gradients or finite differences.

---

## Quick Start

```bash
# Run the testTemplate config
python main.py --config configs/testTemplate.json --out output\resultsTemplate.json
```

**Optional Flags:**
- `--out path.json` → Save results to JSON file
- `--include-x` → Include head/tail preview of optimal point \( x^* \)
- `--keep N` → Keep last N history items (default: 5)

---

## Project Structure
NewtonLab/
├─ configs/
│  └─ testTemplate.json
├─ output/
│  └─ resultsTemplate.json
├─ solvers/
│  ├─ __init__.py
│  ├─ newtonBase.py
│  ├─ modifiedNewton/
│  │  ├─ __init__.py
│  │  ├─ modifiedNewton.py
│  │  └─ solverInstruments.py
│  └─ truncatedNewton/
│     ├─ __init__.py
│     ├─ truncatedNewton.py
│     └─ solverInstruments.py
├─ tools/
│  └─ __init__.py
├─ main.py
├─ README.md
└─ requirements.txt

---

## Configuration

### General Problem Settings
```json
{
  "problem": {
    "function": "extended_rosenbrock",
    "x0": [0, 0, 0, 0, 0, 0, 0, 0],
    "alpha0": 1.0,
    "kmax": 1000,
    "tolgrad": 1e-6,
    "c1": 1e-4,
    "rho": 0.5,
    "btmax": 50,
    "derivatives": "finite_differences",
    "derivative_method": "central",
    "perturbation": 1e-6
  },
  "timing": false,
  "print_every": 50
}
```

**Supported Functions:**
- `extended_rosenbrock`, `extended_powell`
- `broyden_tridiagonal_function`, `rosenbrock`

**Derivative Methods:**
- `exact`, `finite_differences`, `adaptive_finite_differences`
- Methods: `forward`, `backward`, `central` (for finite differences)

### Truncated Newton Settings
```json
{
  "method": "truncated",
  "truncated": {
    "eta": 0.5,
    "rate_of_convergence": "superlinear"
  }
}
```
- `eta`: CG tolerance cap factor
- `rate_of_convergence`: `superlinear` (√‖g‖) or `quadratic` (‖g‖)

### Modified Newton Settings
```json
{
  "method": "modified",
  "modified": {
    "solver_linear_system": "chol",
    "H_correction_factor": 1e-3,
    "precond": "no"
  }
}
```
- `solver_linear_system`: `chol` (Cholesky) or `cg` (Conjugate Gradient)
- `precond`: `yes`/`no` (preconditioning for CG only)

---

## Example Configurations

### Truncated Newton: Rosenbrock (10**2)
```json
{
  "method": "truncated",
  "problem": {
    "function": "rosenbrock",
    "x0": [1.2, 1.2],
    "alpha0": 1.0,
    "kmax": 1000,
    "tolgrad": 1e-6,
    "c1": 1e-4,
    "rho": 0.5,
    "btmax": 50,
    "derivatives": "finite_differences",
    "derivative_method": "central",
    "perturbation": 1e-6
  },
  "truncated": {
    "eta": 0.5,
    "rate_of_convergence": "superlinear"
  },
  "timing": false,
  "print_every": 50
}
```

### Modified Newton: Rosenbrock (10**2)
```json
{
  "method": "modified",
  "problem": {
    "function": "rosenbrock",
    "x0": [1.2, 1.2],
    "alpha0": 1.0,
    "kmax": 1000,
    "tolgrad": 1e-6,
    "c1": 1e-4,
    "rho": 0.5,
    "btmax": 50,
    "derivatives": "finite_differences",
    "derivative_method": "central",
    "perturbation": 1e-6
  },
  "modified": {
    "solver_linear_system": "chol",
    "H_correction_factor": 1e-3,
    "precond": "no"
  },
  "timing": false,
  "print_every": 50
}
```
---

## Output Format

Compact printed summary + optional JSON output containing:

- **Optimization Results:** `method`, `function`, `n`, `k`, `success`
- **Solution Quality:** `f_star`, `grad_norm_final`
- **History Tails:** Last N values of `grad_norm_last`, `bt_last`, `tol_last` (TN), `inner_last`
- **Statistics:** `inner_summary`, `bt_summary`, `grad_norm_summary`, `tol_summary`
- **Performance:** `avg_iter_time_sec`, `total_time_sec`
- **Optional:** `x_star_preview` (with `--include-x` flag)

> Output remains compact even for large-scale problems (e.g., \( n = 10^5 \)).

## Example Output
### Truncated Newton: Rosenbrock (10**2)
```json
  {
    "method": "truncated",
    "function": "rosenbrock",
    "n": 2,
    "k": 9,
    "success": true,
    "f_star": 7.776028393206474e-16,
    "grad_norm_final": 4.200786151495075e-08,
    "grad_norm_last": [
      125.16932530315574,
      9.923850780109342,
      0.1229978649738347,
      1.657644841750929,
      0.0443619388397416,
      1.2858927401645461,
      0.0034223590336830345,
      0.004724272844621177,
      1.3699466582117555e-05,
      4.200786151495075e-08
    ],
    "bt_seq": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "tol_seq": [
      62.58466265157787,
      4.961925390054671,
      0.04313664450614411,
      0.8288224208754645,
      0.009343633134848425,
      0.6429463700822731,
      0.0002002111927818932,
      0.0003247150797619412,
      5.0705548612426565e-08
    ],
    "inner_last": [
      1,
      1,
      2,
      1,
      2,
      1,
      2,
      1,
      2
    ],
    "x_star_preview": {
      "len": 2,
      "head": [
        0.9999999721166472,
        0.9999999441984421
      ],
      "tail": []
    },
    "avg_iter_time_sec": 0.20986074871487087,
    "total_time_sec": 1.888746738433838
  }
```
### Modified Newton: Rosenbrock (10**2)
```json
 {
    "method": "modified",
    "function": "rosenbrock",
    "n": 2,
    "k": 8,
    "success": true,
    "f_star": 3.9953667786252693e-20,
    "grad_norm_final": 1.4285817705391666e-11,
    "grad_norm_last": [
        125.16932530315574,
        0.3998297697447818,
        4.7849660199002315,
        0.6562640686040616,
        1.265932823292802,
        0.034655559898786464,
        0.00801991311893653,
        1.4516788695914186e-06,
        1.4285817705391666e-11
    ],
    "bt_seq": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    ],
    "tol_seq": null,
    "inner_last": [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
    ],
    "x_star_preview": {
        "len": 2,
        "head": [
        0.9999999998001161,
        0.9999999996002007
        ],
        "tail": []
    },
    "avg_iter_time_sec": 0.04942464828491211,
    "total_time_sec": 0.3953971862792969
}
```
---

## Usage Tips

- **Large-scale problems:** Prefer Truncated Newton with finite differences
- **Silent operation:** Set `"print_every": 0` to disable iteration logs
- **Memory efficiency:** Avoid dumping full vectors to JSON; use preview options
- **Performance:** Use exact derivatives when available for faster convergence

---