from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Any
import jax.numpy as jnp
import numpy as np          # only for the final “scatter back” to the python side

from .riemannian_adam import RiemannianAdam
from weyl_compression.circuit.circuit_dataclasses import Circuit
from weyl_compression.optimization.gradient import overlap_and_environments
from weyl_compression.utils.batching import build_buckets, _iterate_canonical
from weyl_compression.optimization.utils import overlap_to_loss

def _scatter_back(
    circuit,
    U_single: jnp.ndarray,
    U_double: jnp.ndarray,
    idx_single: List[int],
    idx_double: List[int],
) -> None:
    """
    Overwrite `gate.matrix` in-place with the freshly updated unitaries.

    Works purely on the Python side; does not touch the optimiser states.
    """
    # Build a flat index  ->  matrix map
    idx2mat: Dict[int, Any] = {}
    for k, i in enumerate(idx_single):
        idx2mat[i] = np.array(U_single[k])   # host copy; small (2×2)
    for k, i in enumerate(idx_double):
        idx2mat[i] = np.array(U_double[k])   # (4×4)

    # Walk the circuit once and assign
    for flat_i, gate in enumerate(_iterate_canonical(circuit)):
        gate.matrix = idx2mat[flat_i]

# NOTE: consider making an object optimizer later on (for checkpointing, schedulers, multiple objectives, multi-optimizer, etc.)
def optimize_riemannian(
    circuit,
    reference_mpo,
    *,
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    clip_grad_norm: float = None,
    max_steps: int = 1000,
    callback: Callable = None,
    init_vertical_sweep = "top-down",
    max_bondim_env : int,
    svd_cutoff: float = 1e-12,
) -> Tuple[Circuit, List[float]]:
    """
    Train the circuit in-place and return it together with the list of loss values.

    A user-supplied callback is invoked at every step with keyword arguments
    `step`, `loss`.
    """

    U1, U2, idx1, idx2 = build_buckets(circuit)

    opt1, opt2 = RiemannianAdam(lr, betas, eps, clip_grad_norm), \
                 RiemannianAdam(lr, betas, eps, clip_grad_norm)
    state1, state2 = opt1.init(U1), opt2.init(U2)

    def _vert_dir(step: int) -> str:
        """alternate bottom-up / top-down every iteration"""

        if init_vertical_sweep not in ('bottom-up', 'top-down'):
            raise ValueError("vertical_sweep must be one of: 'bottom-up', 'top-down'")
        even_dir  = init_vertical_sweep
        odd_dir   = "bottom-up" if even_dir == "top-down" else "top-down"
        return even_dir if (step % 2 == 0) else odd_dir

    history: List[float] = []

    for it in range(max_steps):
        overlap, grads_ordered, _ = overlap_and_environments(
            circuit,
            reference_mpo,
            vertical_sweep=_vert_dir(it),
            max_bondim_env=max_bondim_env,
            svd_cutoff=svd_cutoff,
        )
        # compute grad of HST loss function using trace gradients
        # L = 1 - c |T|^2,      T = overlap = <Env, G>
        # dL/dG = -2 c conj(T) * dT/dG  with dT/dG = Env (no conjugation)
        if reference_mpo.is_normalized:
            c = 1.0 / (2 ** circuit.n_sites)
        else:
            c = 1.0 / (2 ** (2 * circuit.n_sites))

        scale = -2.0 * c * overlap

        # convert "trace partial deriv wrt G" dT/dG_i into HST grads: dL/dG_i
        grads_ordered = [scale * g.conj() for g in grads_ordered]

        # compute loss
        loss = overlap_to_loss(overlap=overlap, kind='HST', n_sites=circuit.n_sites, normalize=reference_mpo.is_normalized) 
        # 1.2 split gradients into the two homogeneous stacks
        print(f"It: {it}, Loss: {loss}")

        g1 = (
            jnp.stack([grads_ordered[i] for i in idx1], axis=0)
            if idx1 else jnp.empty((0, 2, 2), dtype=circuit.dtype)
        )
        g2 = (
            jnp.stack([grads_ordered[i] for i in idx2], axis=0)
            if idx2 else jnp.empty((0, 4, 4), dtype=circuit.dtype)
        )

        # 1.3 optimiser steps
        U1, state1, _ = opt1.step(U1, g1, state1)
        U2, state2, _ = opt2.step(U2, g2, state2)

        # 1.4 write updated unitaries back to the live circuit
        _scatter_back(circuit, U1, U2, idx1, idx2)

        # 1.5 bookkeeping
        history.append(loss)

        if callback is not None:
            should_stop = callback(
                step=it,
                loss=loss,
            )
            if should_stop:
                return circuit, history

    return circuit, history
