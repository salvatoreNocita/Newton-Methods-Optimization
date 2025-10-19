from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

# Se hai il package strutturato come suggerito:
# your_project/
# └─ optim/
#    ├─ __init__.py (re-esporta TruncatedNewton, ModifiedNewton)
#    ├─ newton_base.py
#    ├─ truncated_newton.py
#    └─ modified_newton.py
try:
    from solvers.modifiedNewton import modifiedNewton
    from solvers.truncatedNewton import truncatedNewton
except Exception:
    # fallback se import relativo non funziona: prova ad aggiungere la root
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from solvers.modifiedNewton import modifiedNewton
    from solvers.truncatedNewton import truncatedNewton


def _np_array_from(obj: Any, name: str) -> np.ndarray:
    """Converte una lista JSON in np.ndarray (float)."""
    if isinstance(obj, list):
        return np.asarray(obj, dtype=float)
    raise TypeError(f"'{name}' deve essere una lista di numeri nel JSON.")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _preview_1d(seq, keep: int = 5):
    # returns {"len": ..., "head": [...], "tail": [...]}
    if seq is None:
        return None
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()
    n = len(seq)
    if n == 0:
        return {"len": 0, "head": [], "tail": []}
    head = seq[:keep]
    tail = seq[-keep:] if n > keep else []
    return {"len": n, "head": head, "tail": tail}


def _run_single_test(cfg, include_x : bool, keep : int) -> Dict[str, Any]:
    """
    Esegue un singolo test preso dal JSON e ritorna un dizionario con i risultati principali.
    Struttura attesa (minima):
    {
      "method": "truncated" | "modified",
      "problem": {
        "function": "...",
        "x0": [...],
        "alpha0": 1.0, "kmax": 1000, "tolgrad": 1e-6,
        "c1": 1e-4, "rho": 0.5, "btmax": 50,
        "derivatives": "exact" | "finite_differences" | "adaptive_finite_differences",
        "derivative_method": "forward" | "backward" | "central",
        "perturbation": 1e-6
      },
      "truncated": {
        "eta": 0.5,
        "rate_of_convergence": "superlinear"
      },
      "modified": {
        "solver_linear_system": "chol",
        "H_correction_factor": 1e-3,
        "precond": "no"
      },
      "timing": false,
      "print_every": 50
    }
    """
    method = cfg.get("method", "").strip().lower()
    problem = cfg.get("problem", {})
    if not method:
        raise ValueError("Campo 'method' mancante nel JSON.")
    if not problem:
        raise ValueError("Sezione 'problem' mancante nel JSON.")

    # ---- Base kwargs (comuni alla NewtonBase) ----
    x0 = _np_array_from(problem["x0"], "problem.x0")
    base_kwargs = dict(
        x0=x0,
        function=problem["function"],
        alpha0=float(problem["alpha0"]),
        kmax=int(problem["kmax"]),
        tolgrad=float(problem["tolgrad"]),
        c1=float(problem["c1"]),
        rho=float(problem["rho"]),
        btmax=int(problem["btmax"]),
        derivatives=problem["derivatives"],
        derivative_method=problem["derivative_method"],
        perturbation=float(problem["perturbation"]),
    )

    # timing e print_every (opzionali)
    timing = bool(cfg.get("timing", False))
    print_every = int(cfg.get("print_every", 50))

    # ---- Istanziazione per metodo scelto ----
    if method == "truncated":
        tsec = cfg.get("truncated", {})
        eta = float(tsec["eta"])
        # rate_of_convergence qui vive nella figlia (come consigliato)
        roc = tsec.get("rate_of_convergence", "superlinear")

        solver = truncatedNewton.truncatedNewton(
            eta=eta,
            rate_of_convergence=roc,
            **base_kwargs,
        )

    elif method == "modified":
        msec = cfg.get("modified", {})
        solver = modifiedNewton.modifiedNewton(
            solver_linear_system=msec["solver_linear_system"],
            H_correction_factor=float(msec["H_correction_factor"]),
            precond=msec["precond"],
            **base_kwargs,
        )
    else:
        raise ValueError("Valore di 'method' non valido. Usa 'truncated' o 'modified'.")

    # ---- Esecuzione ----
    exec_times, x_star, f_star, grad_norm_seq, k, success, inner_iters, bt_seq, tol_seq = solver.Run(
        timing=timing, print_every=print_every
    )

    # ---- Output compatto per stampa/salvataggio ----
    out = {
        "method": method,
        "function": problem["function"],
        "n": int(x0.size),
        "k": int(k),
        "success": bool(success),
        "f_star": float(f_star),
        "grad_norm_final": float(grad_norm_seq[-1] if grad_norm_seq else float("nan")),
        # histories: only keep last `keep` elements (compact, useful)
        "grad_norm_last": grad_norm_seq[-keep:] if grad_norm_seq else [],
        "bt_seq": bt_seq[-keep:] if bt_seq else [],
        "tol_seq": (tol_seq[-keep:] if tol_seq else None),
        "inner_last": inner_iters[-keep:] if inner_iters else [],
        # x*: include only preview if requested, otherwise omit entirely
        "x_star_preview": _preview_1d(x_star, keep) if include_x else None,
        # timing summary
        "avg_iter_time_sec": float(np.mean(exec_times) if exec_times else 0.0),
        "total_time_sec": float(np.sum(exec_times) if exec_times else 0.0),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description="Esegui Truncated/Modified Newton da un file JSON di configurazione.")
    parser.add_argument("--config", required=True, help="Path al file JSON di configurazione.")
    parser.add_argument("--out", default=None, help="(Opz.) Path file JSON per salvare i risultati.")
    parser.add_argument("--include-x", action="store_true",
                    help="Include only a head/tail preview of x* in the JSON")
    parser.add_argument("--keep", type=int, default=5,
                    help="How many items to keep in head/tail previews (default: 5)")
    args = parser.parse_args()

    cfg = _load_json(args.config)

    # Supporta sia un singolo test che una lista di test
    if isinstance(cfg, list):
        results = [_run_single_test(test, include_x=args.include_x, keep=args.keep) for test in cfg]
    elif isinstance(cfg, dict) and "tests" in cfg:
        results = [_run_single_test(test, include_x=args.include_x, keep=args.keep) for test in cfg["tests"]]
    else:
        results = [_run_single_test(cfg, include_x=args.include_x, keep=args.keep)]

    # --- Stampa riepilogo ---
    for i, r in enumerate(results, 1):
        print("=" * 60)
        print(f"[{i}] method={r['method']} | func={r['function']} | n={r['n']}")
        print(f"   success={r['success']} | k={r['k']}")
        print(f"   f*={r['f_star']:.6e} | ||grad||={r['grad_norm_final']:.3e}")
        print(f"   time: avg={r['avg_iter_time_sec']:.4f}s  total={r['total_time_sec']:.4f}s")
        if r["tol_seq"] is not None:
            print(f"   (TN) cg_tols: first={r['tol_seq'][0]:.3e} last={r['tol_seq'][-1]:.3e}")

    # --- Salvataggio su file se richiesto ---
    if args.out:
        payload = results if len(results) > 1 else results[0]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("-" * 60)
        print(f"Risultati salvati in: {args.out}")


if __name__ == "__main__":
    main()
