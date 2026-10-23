from __future__ import annotations
import weyl_compression.jax_config

from typing import Dict, List, Tuple, Any
import jax.numpy as jnp

# NOTE: pytree technique might be slower, since we are making each parameter a leaf, i.e. an array. 
        # It might be smarter to have a whole array for at least each layer, thereby grouping the parameters.

# ---- Type aliases: the parameter PyTree and a parallel "meta" tree ----
# Each leaf is a 1D jnp.ndarray of angles for that gate (possibly length 0 for fixed gates).
ParamsTree = Dict[int, Dict[int, jnp.ndarray]]     # layer_index -> gate_index -> theta (1D)
MetaTree   = Dict[int, Dict[int, dict]]            # same keys; small dict with metadata per gate

def _flatten_gate_params_to_vec(params_tuple: Tuple[Any, ...],
                                dtype=jnp.float64) -> Tuple[jnp.ndarray, List[int]]:
    """
    Pack Gate.params (tuple of scalars/arrays) into a single 1D vector (jnp.ndarray).
    Returns (theta, splits) where 'splits' records the raveled sizes of each element.

    If the gate is not parameterized (empty tuple), returns a length-0 vector and splits=[].
    """
    if not params_tuple:
        return jnp.zeros((0,), dtype=dtype), []

    chunks: List[jnp.ndarray] = []
    splits: List[int] = []

    for p in params_tuple:
        a = jnp.asarray(p, dtype=dtype).ravel()
        chunks.append(a)
        splits.append(int(a.size))

    theta = jnp.concatenate(chunks) if chunks else jnp.zeros((0,), dtype=dtype)
    return theta, splits

def _unflatten_vec_to_gate_params(theta: jnp.ndarray, splits: List[int]) -> Tuple[jnp.ndarray, ...]:
    """
    Inverse of _flatten_gate_params_to_vec when you only need a flat tuple of 1D arrays.
    (You can post-process each piece to your preferred shape if needed.)
    """
    if not splits:
        return tuple()
    out: List[jnp.ndarray] = []
    start = 0
    for s in splits:
        out.append(theta[start:start + s])
        start += s
    return tuple(out)

# ------------------------- building the PyTrees -------------------------

def extract_params_tree(circ) -> Tuple[ParamsTree, MetaTree]:
    """
    Build a params PyTree for *all* gates in the circuit and a parallel meta tree.
    - Keys: params[layer_idx][gate_idx] = 1D jnp.ndarray of angles (len 0 if no params)
    - Meta keeps: {'arity': 1 or 2, 'name': str, 'qubits': tuple, 'splits': List[int]}

    This is stable because it uses layer.iterate_gates() which sorts gates spatially.
    """
    circ.sort_layers()
    params_tree: ParamsTree = {}
    meta_tree: MetaTree = {}

    for layer in circ.layers:
        layer_params: Dict[int, jnp.ndarray] = {}
        layer_meta: Dict[int, dict] = {}

        for gi, g in enumerate(layer.iterate_gates()):
            theta, splits = _flatten_gate_params_to_vec(g.params, dtype=jnp.float64)
            layer_params[gi] = theta
            layer_meta[gi] = {
                "arity": len(g.qubits),
                "name": g.name,
                "qubits": tuple(g.qubits),
                "splits": splits,  # for potential reconstruction
            }

        params_tree[layer.layer_index] = layer_params
        meta_tree[layer.layer_index] = layer_meta

    return params_tree, meta_tree
