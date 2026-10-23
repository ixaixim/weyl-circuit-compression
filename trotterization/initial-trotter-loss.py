"""
Loss of the unoptimized Trotter circuits (the initial ansatz of run-optimization.py) against the
reference MPO, for several numbers of Trotter repetitions. These are the dashed "initial" curves
of the comparison figures and equal loss_history[0] of the corresponding optimization runs.

Usage: python initial-trotter-loss.py <config.yml> --reps 1 2 3 [--reference PATH] [--out JSON]
"""
import weyl_compression.jax_config

import argparse
import json
import time
from pathlib import Path

from weyl_compression.experiment import build_ansatz, circuit_loss, load_config, native_depth, reference_filename
from weyl_compression.mpo.mpo_dataclass import MPO

RUN_DIR = Path(__file__).resolve().parents[1] / "run"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--reps", type=int, nargs="+", required=True, help="Trotter repetitions to evaluate")
    parser.add_argument("--reference", type=Path, default=None,
                        help="reference MPO file (default: the one described by the config, in run/<model>/reference/)")
    parser.add_argument("--out", type=Path, default=None, help="write the results to this JSON file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reference_path = args.reference or RUN_DIR / cfg["model"] / "reference" / reference_filename(cfg)
    reference = MPO.load_pickle(reference_path)
    max_bondim_env, svd_cutoff = cfg["optimizer"]["max_bondim_env"], cfg["optimizer"]["svd_cutoff"]

    points = []
    for reps in args.reps:
        start = time.time()
        circuit = build_ansatz(cfg, reps)
        loss = circuit_loss(circuit, reference, max_bondim_env, svd_cutoff)
        points.append({
            "reps": reps,
            "num_2q_layers": circuit.num_2q_layers,
            "native_depth": native_depth(cfg["model"], cfg["method"], reps),
            "initial_loss": loss,
            "seconds": round(time.time() - start, 1),
        })
        print(f"reps={reps:3d}  native depth={points[-1]['native_depth']:3d}  initial loss={loss:.16e}")

    if args.out:
        result = {
            "model": cfg["model"], "method": cfg["method"], "n_sites": cfg["n_sites"], "t": cfg["t"],
            "hamiltonian": cfg["hamiltonian"], "ansatz_order": cfg["ansatz"]["order"],
            "reference": reference_path.name, "max_bondim_env": max_bondim_env, "svd_cutoff": svd_cutoff,
            "points": points,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
