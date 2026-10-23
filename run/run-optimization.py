"""
Compress a Trotterized time-evolution circuit by optimizing it against the reference MPO
of a config (generate the reference first with generate-reference.py).

- method 'riemannian':     Riemannian Adam over generic two-qubit unitaries (brickwall).
- method 'hardware-aware': Adam over the angles of native two-qubit gates (R_ZZ for Ising,
                           fSim for Heisenberg); single-qubit gates by Riemannian Adam.

The run (loss history, optimized circuit, full config) is saved as a pickled dictionary (.pkl) in
run/<model>/results/.

Usage: python run-optimization.py <config.yml> [--reps R] [--max-steps S] [--out-dir DIR]
"""
import weyl_compression.jax_config

import argparse
import time
from pathlib import Path

from weyl_compression.experiment import (
    build_ansatz, circuit_loss, load_config, make_early_stop, reference_filename, save_run,
)
from weyl_compression.mpo.mpo_dataclass import MPO
from weyl_compression.optimization.hardware_aware.fsim import optimize_fsim
from weyl_compression.optimization.hardware_aware.rzz import optimize_rzz
from weyl_compression.optimization.riemannian_adam.optimizer import optimize_riemannian
from weyl_compression.optimization.schedulers import ReduceLROnPlateau

RUN_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--reps", type=int, default=None, help="override ansatz.reps (Trotter repetitions)")
    parser.add_argument("--max-steps", type=int, default=None, help="override optimizer.max_steps")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory (default: run/<model>/results/)")
    args = parser.parse_args()

    cfg = load_config(args.config, reps=args.reps, max_steps=args.max_steps)
    model_dir = RUN_DIR / cfg["model"]

    reference_path = model_dir / "reference" / reference_filename(cfg)
    if not reference_path.exists():
        raise SystemExit(f"Reference MPO not found at {reference_path}. "
                         f"Run: python generate-reference.py {args.config}")
    reference = MPO.load_pickle(reference_path)
    print(f"Loaded reference MPO from {reference_path}")

    circuit = build_ansatz(cfg)
    print(f"{cfg['model']} / {cfg['method']}: n_sites={cfg['n_sites']}, t={cfg['t']}, "
          f"reps={cfg['ansatz']['reps']}, {circuit.num_2q_layers} two-qubit layers")

    early_stop = make_early_stop(**cfg["early_stopping"])
    start = time.time()
    if cfg["method"] == "riemannian":
        circuit, loss_history = optimize_riemannian(circuit, reference, callback=early_stop, **cfg["optimizer"])
    else:
        optimize = optimize_rzz if cfg["model"] == "ising" else optimize_fsim
        circuit, loss_history = optimize(
            circuit, reference, callback=early_stop, scheduler=ReduceLROnPlateau(**cfg["scheduler"]),
            **cfg["optimizer"],
        )
    runtime_seconds = time.time() - start

    final_circuit_loss = circuit_loss(
        circuit, reference, cfg["optimizer"]["max_bondim_env"], cfg["optimizer"]["svd_cutoff"],
    )
    print(f"Initial loss {float(loss_history[0]):.6e}, final loss {final_circuit_loss:.6e} "
          f"after {len(loss_history)} steps ({runtime_seconds:.0f} s)")

    save_run(
        args.out_dir or model_dir / "results", cfg, circuit, loss_history,
        reference_path=reference_path, final_circuit_loss=final_circuit_loss, runtime_seconds=runtime_seconds,
    )


if __name__ == "__main__":
    main()
