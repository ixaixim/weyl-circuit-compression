# utils/batching.py
from __future__ import annotations

from typing import List, Tuple
import jax.numpy as jnp

from weyl_compression.circuit.circuit_dataclasses import Circuit


# ---------------------------------------------------------------------------
# 1.  Canonical iteration helper
# ---------------------------------------------------------------------------
def _iterate_canonical(circ: Circuit):
    """
    Yield gates in the same canonical order used by
    `sweeping_euclidean_gradient_*`:

        layer index ascending
        └── within a layer: qubits ascending  (left → right)
    """
    for l in range(circ.num_layers):
        layer = circ.layers[l]
        # iterate_gates(reverse=False) already sorts spatially
        for gate in layer.iterate_gates(reverse=False):
            yield gate


# ---------------------------------------------------------------------------
# 2.  Main entry point
# ---------------------------------------------------------------------------
def build_buckets(
    circ: Circuit,
) -> Tuple[jnp.ndarray, jnp.ndarray, List[int], List[int]]:
    """
    Scan `circ` once and return

        U_single : (G1, 2, 2)  stacked 1-qubit unitary blocks
        U_double : (G2, 4, 4)  stacked 2-qubit unitary blocks
        idx_single, idx_double : positions of those gates
                                 in the **flat canonical order**

    The index arrays let you `take()` the matching gradients and later
    `scatter` the updated unitaries back into the circuit.
    """
    # -- buckets -------------------------------------------------------------
    single_idx:  List[int] = []
    double_idx:  List[int] = []
    mats_single: List[jnp.ndarray] = []
    mats_double: List[jnp.ndarray] = []

    for flat_i, gate in enumerate(_iterate_canonical(circ)):
        mat = jnp.asarray(gate.matrix, dtype=circ.dtype)

        if gate.is_single_qubit():
            single_idx.append(flat_i)
            mats_single.append(mat)           # (2,2)
        elif gate.is_two_qubit():
            double_idx.append(flat_i)
            mats_double.append(mat)           # (4,4)
        else:
            raise ValueError(f"Unsupported gate with {len(gate.qubits)} qubits.")

    # -- stack (handle empty buckets gracefully) ----------------------------
    if mats_single:
        U_single = jnp.stack(mats_single, axis=0)         # (G1,2,2)
    else:
        U_single = jnp.empty((0, 2, 2), dtype=circ.dtype)

    if mats_double:
        U_double = jnp.stack(mats_double, axis=0)         # (G2,4,4)
    else:
        U_double = jnp.empty((0, 4, 4), dtype=circ.dtype)

    return U_single, U_double, single_idx, double_idx
