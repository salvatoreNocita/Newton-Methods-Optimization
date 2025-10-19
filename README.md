# NewtonLab — Truncated & Modified Newton

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
```
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
```
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
Fields' meanings:
* **`function`**: which test objective to minimize. Supported:
  `extended_rosenbrock`, `extended_powell`, `broyden_tridiagonal_function`, `rosenbrock`.
* **`x0`**: initial point (list of numbers). Its length = problem dimension `n`.
* **`alpha0`**: initial step size for line search (typically `1.0` for Newton-type methods).
* **`kmax`**: maximum number of outer iterations (safeguard cap).
* **`tolgrad`**: stopping tolerance on the gradient norm, i.e., stop when `||∇f(x)|| ≤ tolgrad`.
* **`c1`**: Armijo parameter in backtracking (sufficient decrease). Usual range `1e-4`–`1e-2`.
* **`rho`**: backtracking shrink factor for the step size, in `(0,1)` (e.g., `0.5` halves `α` each backtrack).
* **`btmax`**: max number of backtracking reductions per iteration.
* **`derivatives`**: how to compute gradient/Hessian info:
  * `"exact"` → use analytic derivatives where available.
  * `"finite_differences"` → numerical FD (fixed step).
  * `"adaptive_finite_differences"` → numerical FD with per-coordinate step.
* **`derivative_method`** (used when FD is selected):
  `"forward"`, `"backward"`, or `"central"` (central is typically most accurate but costs ~2×).
* **`perturbation`**: FD step size `h` (e.g., `1e-6`). If using adaptive FD, this is the base scale for building per-coordinate steps.

Top-level controls:
* **`timing`**: if `true`, enforces a time budget per run using a heuristic max-time; if `false`, no time cap (runs until convergence/`kmax`).
* **`print_every`**: how often to print iteration logs (e.g., `50` → print every 50 iterations; `0` to silence).

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
Fields' meanings:
* **`method`**: `"truncated"` → use Truncated Newton.
* **`truncated.eta`**: cap for CG tolerance; smaller = more accurate (more CG iters), larger = cheaper (fewer iters). Typical 0.1–0.9.
* **`truncated.rate_of_convergence`**:
  * `"superlinear"` → ηₖ = √‖g‖ (loose early, tight near solution).
  * `"quadratic"` → ηₖ = ‖g‖ (stricter; targets quadratic local rate).


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

Fields' meanings:
* **`method`**: `"modified"` → use Modified Newton.
* **`modified.solver_linear_system`**: how to solve (H p = -g)
  * `"chol"`: Cholesky (fast for small/medium dense PD (H)).
  * `"cg"`: Conjugate Gradient (better for large/sparse (H)).
* **`modified.H_correction_factor`**: factor used to **add diagonal shifts** until (H) is **positive definite** (larger ⇒ more aggressive/safer but can distort (H) more).
* **`modified.precond`** (only for `"cg"`): `"yes"`/`"no"` to use a preconditioner (e.g., incomplete Cholesky) to speed up CG. Ignored for `"chol"`.

---

## Example Configurations

### Truncated Newton: Rosenbrock (n = 2)
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

### Modified Newton: Rosenbrock (n = 2)
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

## Example Output Report
### Truncated Newton: Rosenbrock (n = 2)
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
---

### Modified Newton: Rosenbrock (n = 2)
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
Here’s what each field means in that **result JSON**:
* **`method`**: solver used — `truncated` = Truncated Newton.
* **`function`**: objective minimized — here, classic `rosenbrock`.
* **`n`**: problem dimension (size of `x`), here 2.
* **`k`**: number of outer iterations performed (9).
* **`success`**: `true` if stop criteria met (e.g., ‖∇f‖ ≤ `tolgrad`) before limits.
* **`f_star`**: final objective value (f(x^*)).
* **`grad_norm_final`**: final gradient norm (|\nabla f(x^*)|).
* **`grad_norm_last`**: tail of the gradient-norm history (here you stored the last 10 values).
* **`bt_seq`**: step sizes accepted by backtracking at each iteration (often 1.0 for Newton).
* **`tol_seq`**: CG tolerances used at each TN iteration (the target residual norms).
* **`inner_last`**: inner CG iterations per outer step (tail only).
* **`x_star_preview`**: small preview of the final solution vector:
  * `len`: dimension of (x^*)
  * `head` / `tail`: first/last few entries (you chose to include only head).
* **`avg_iter_time_sec`**: average time per outer iteration (seconds).
* **`total_time_sec`**: total solver runtime (seconds).

> Output remains compact even for large-scale problems (e.g., \( n = 10^5 \)).

---

## Usage Tips

- **Large-scale problems:** Prefer Truncated Newton with finite differences
- **Silent operation:** Set `"print_every": 0` to disable iteration logs
- **Memory efficiency:** Avoid dumping full vectors to JSON; use preview options
- **Performance:** Use exact derivatives when available for faster convergence

---