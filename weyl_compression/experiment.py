"""
Shared set-up for the paper experiments: configuration loading, reference naming,
initial (Trotterized) ansatz construction, loss evaluation and run serialization.

Used by `run/`, `trotterization/` and `plotting-results/`, so that all of them build
circuits and evaluate losses in exactly the same way.
"""
from __future__ import annotations

import weyl_compression.jax_config  # enables float64 before any jax array is created

import datetime
import hashlib
import pickle
import platform
import subprocess
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from weyl_compression.circuit.circuit_dataclasses import Circuit
from weyl_compression.circuit.trotter.even_odd import even_odd_trotter_circuit
from weyl_compression.circuit.trotter.hardware_aware_ising import ising_hardware_aware_trotter_circuit
from weyl_compression.circuit.trotter.hardware_aware_heisenberg import heisenberg_hardware_aware_trotter_circuit
from weyl_compression.circuit.native_gates.rzz import parameterize_rzz_circuit
from weyl_compression.circuit.native_gates.fsim import parameterize_fsim_circuit
from weyl_compression.mpo.mpo_dataclass import MPO
from weyl_compression.optimization.gradient import overlap_and_environments
from weyl_compression.optimization.utils import overlap_to_loss

MODELS = ("ising", "heisenberg")
METHODS = ("riemannian", "hardware-aware")
HAMILTONIAN_KEYS = ("Jx", "Jy", "Jz", "hx", "hy", "hz")
_SECTIONS = {
    "hamiltonian": set(HAMILTONIAN_KEYS),
    "reference": {"reps", "order", "max_bond_dim", "compress", "err_threshold"},
    "ansatz": {"reps", "order"},
    "early_stopping": {"patience", "min_delta"},
}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def _coerce_numbers(obj: Any) -> Any:
    """PyYAML (YAML 1.1) reads `1e-4` as a string; turn such strings into floats."""
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return obj
    return obj


def load_config(path: str | Path, reps: int | None = None, max_steps: int | None = None) -> dict[str, Any]:
    """
    Load and validate an experiment configuration (YAML).

    `reps` and `max_steps` override `ansatz.reps` and `optimizer.max_steps`.
    The `optimizer` (and `scheduler`) blocks are passed verbatim as keyword arguments to the
    optimizer, so unknown keys there fail at the call site.
    """
    import yaml

    cfg = _coerce_numbers(yaml.safe_load(Path(path).read_text()))

    if cfg.get("model") not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {cfg.get('model')!r}")
    if cfg.get("method") not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {cfg.get('method')!r}")

    expected = {"model", "method", "n_sites", "t", "optimizer", *_SECTIONS}
    if cfg["method"] == "hardware-aware":
        expected.add("scheduler")
    if set(cfg) != expected:
        raise ValueError(f"config keys {sorted(cfg)} differ from expected {sorted(expected)}")
    for section, keys in _SECTIONS.items():
        if set(cfg[section]) != keys:
            raise ValueError(f"'{section}' keys {sorted(cfg[section])} differ from expected {sorted(keys)}")

    # Floats keep reference file names stable (e.g. 'Jx0.0' rather than 'Jx0').
    cfg["t"] = float(cfg["t"])
    cfg["hamiltonian"] = {k: float(v) for k, v in cfg["hamiltonian"].items()}
    cfg["optimizer"]["betas"] = tuple(cfg["optimizer"]["betas"])

    if reps is not None:
        cfg["ansatz"]["reps"] = reps
    if max_steps is not None:
        cfg["optimizer"]["max_steps"] = max_steps
    return cfg


def reference_filename(cfg: dict[str, Any]) -> str:
    """File name of the reference MPO described by `cfg` (bond dimension included)."""
    h, ref = cfg["hamiltonian"], cfg["reference"]
    name = (f"ref_n{cfg['n_sites']}_Jx{h['Jx']}_Jy{h['Jy']}_Jz{h['Jz']}_hx{h['hx']}_hy{h['hy']}_hz{h['hz']}_"
            f"t{cfg['t']}_reps{ref['reps']}_order{ref['order']}_chi{ref['max_bond_dim']}")
    if ref["compress"]:
        name += f"_tol{ref['err_threshold']:g}"
    return name + ".pkl"


# -----------------------------------------------------------------------------
# Circuits and loss
# -----------------------------------------------------------------------------

def build_ansatz(cfg: dict[str, Any], reps: int | None = None) -> Circuit:
    """
    Initial circuit of the optimization: a Trotterized time evolution over time `t`.

    - riemannian:     even-odd brickwall of generic two-qubit gates (field terms split
                      between the two layers).
    - hardware-aware: native-gate splitting, decomposed into parameterized R_ZZ (Ising)
                      or fSim (Heisenberg) gates and single-qubit layers.
    """
    h = cfg["hamiltonian"]
    n_sites, t = cfg["n_sites"], cfg["t"]
    reps = cfg["ansatz"]["reps"] if reps is None else reps
    order = cfg["ansatz"]["order"]
    dt = t / reps

    if cfg["method"] == "riemannian":
        return even_odd_trotter_circuit(
            n_sites=n_sites, Jx=h["Jx"], Jy=h["Jy"], Jz=h["Jz"], hx=h["hx"], hy=h["hy"], hz=h["hz"],
            dt=dt, reps=reps, order=order, method="suzuki", dtype=jnp.complex128,
        )

    if cfg["model"] == "ising":
        if h["Jx"] != 0 or h["Jy"] != 0 or h["hy"] != 0:
            raise ValueError("The hardware-aware Ising ansatz requires Jx = Jy = hy = 0.")
        circ = ising_hardware_aware_trotter_circuit(
            n_sites=n_sites, J=h["Jz"], hx=h["hx"], hz=h["hz"], dt=dt, reps=reps, order=order,
            dtype=jnp.complex128,
        )
        return parameterize_rzz_circuit(circ)

    if h["Jx"] != h["Jy"]:
        raise ValueError("The hardware-aware Heisenberg (XXZ) ansatz requires Jx = Jy.")
    circ = heisenberg_hardware_aware_trotter_circuit(
        n_sites=n_sites, J=h["Jx"], Delta=h["Jz"] / h["Jx"], hx=h["hx"], hy=h["hy"], hz=h["hz"],
        dt=dt, reps=reps, order=order, dtype=jnp.complex128,
    )
    return parameterize_fsim_circuit(circ)


def native_depth(model: str, method: str, reps: int) -> int:
    """
    Native entangling-gate depth of a second-order Trotter ansatz with `reps` repetitions
    (x-axis of the comparison figures). Riemannian circuits count the synthesized depth:
    3 R_ZZ per generic gate (Ising), fSim + R_ZZ per generic gate (Heisenberg).
    """
    if method == "riemannian":
        return (3 if model == "ising" else 2) * (2 * reps + 1)
    return 2 * reps + 2 if model == "ising" else 3 * reps + 1


def circuit_loss(circuit: Circuit, reference: MPO, max_bondim_env: int, svd_cutoff: float) -> float:
    """
    HST loss 1 - |Tr(U_ref^dag W)|^2 / 2^N, contracted exactly as in the first optimizer step
    (top-down sweep), so it reproduces `loss_history[0]` of a run.
    """
    overlap, _, _ = overlap_and_environments(
        circuit, reference, max_bondim_env=max_bondim_env, svd_cutoff=svd_cutoff, vertical_sweep="top-down",
    )
    return float(overlap_to_loss(overlap, kind="HST", n_sites=circuit.n_sites, normalize=reference.is_normalized))


def make_early_stop(patience: int, min_delta: float) -> Callable[..., bool]:
    """Callback stopping once the loss has not improved by `min_delta` for `patience` steps."""
    best_loss = np.inf
    stalled_steps = 0

    def early_stop(*, step: int, loss: float, **_) -> bool:
        nonlocal best_loss, stalled_steps
        if loss < best_loss - min_delta:
            best_loss = loss
            stalled_steps = 0
        else:
            stalled_steps += 1
        if stalled_steps >= patience:
            print(f"Early stopping at step {step}")
            return True
        return False

    return early_stop


# -----------------------------------------------------------------------------
# Run files
# -----------------------------------------------------------------------------

def _serializable(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, jax.Array)):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    return obj


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_run(
    out_dir: Path,
    cfg: dict[str, Any],
    circuit: Circuit,
    loss_history: list[float],
    *,
    reference_path: Path,
    final_circuit_loss: float,
    runtime_seconds: float,
) -> Path:
    """
    Save an optimization run (a pickled dictionary) to `out_dir/run_<model>_<method>_n<N>_r<reps>_<timestamp>.pkl`.

    `loss_history[i]` is the loss before update i; `final_circuit_loss` is the loss of the
    returned (saved) circuit.
    """
    import qiskit

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{cfg['model']}_{cfg['method']}_n{cfg['n_sites']}_r{cfg['ansatz']['reps']}_{timestamp}.pkl"

    payload = {
        "model": cfg["model"],
        "method": cfg["method"],
        "final_loss": loss_history[-1],
        "final_circuit_loss": final_circuit_loss,
        "num_2q_layers": circuit.num_2q_layers,
        "hamiltonian_params": {"n_sites": cfg["n_sites"], **cfg["hamiltonian"], "t": cfg["t"]},
        "trotter_params": {"reps_ansatz": cfg["ansatz"]["reps"], "order_ansatz": cfg["ansatz"]["order"]},
        "config": cfg,
        "reference": {"file": reference_path.name, "sha256": _sha256(reference_path)},
        "loss_history": loss_history,
        "runtime_seconds": runtime_seconds,
        "timestamp": timestamp,
        "git_commit": _git_commit(),
        "versions": {"python": platform.python_version(), "jax": jax.__version__,
                     "numpy": np.__version__, "qiskit": qiskit.__version__},
        "circuit": circuit.to_dict(),
    }
    with path.open("wb") as fh:
        pickle.dump(_serializable(payload), fh)
    print(f"Saved run to {path}")
    return path
