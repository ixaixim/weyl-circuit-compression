import weyl_compression.jax_config
import jax.numpy as jnp
from typing import Callable, List, Optional

from weyl_compression.circuit.circuit_dataclasses import Circuit
from weyl_compression.mpo.mpo_dataclass import MPO
from weyl_compression.optimization.gradient import overlap_and_environments
from weyl_compression.optimization.param_adam.adam import Adam
from weyl_compression.optimization.riemannian_adam.riemannian_adam import RiemannianAdam
from weyl_compression.utils.pytree import extract_params_tree, _unflatten_vec_to_gate_params
from weyl_compression.optimization.utils import overlap_to_loss
from weyl_compression.utils.batching import build_buckets
from weyl_compression.optimization.utils import hst_chain_rule


def _rzz_matrix(theta: float, dtype=jnp.complex128):
    """
    RZZ(theta) = exp(-i * theta/2 * Z⊗Z)
    Diagonal elements: exp(-it/2), exp(it/2), exp(it/2), exp(-it/2)
    """
    # theta is a scalar array or float
    phase_m = jnp.exp(-1j * theta / 2)
    phase_p = jnp.exp(1j * theta / 2)
    # diag: [m, p, p, m]
    return jnp.diag(jnp.array([phase_m, phase_p, phase_p, phase_m], dtype=dtype))

def _drzz_matrix(theta: float, dtype=jnp.complex128):
    """
    Derivative of RZZ(theta) w.r.t theta.
    d/dtheta exp(-i theta/2 * ZZ) = -i/2 * ZZ * exp(...)
    Since RZZ is diagonal, this is element-wise mult by diag(-i/2, i/2, i/2, -i/2) 
    So we multilpy the diagonal elements by [-i/2, i/2, i/2, -i/2].
    """
    rzz = _rzz_matrix(theta, dtype)
    factors = jnp.array([-0.5j, 0.5j, 0.5j, -0.5j], dtype=dtype)
    return jnp.diag(factors) @ rzz


def rzz_param_grad(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    """
    Gradient for an RZZ gate.
    Params: 1 (angle).
    U = RZZ(angle).
    """
    dtype = dL_dG.dtype
    
    p_rzz = theta[0]
    
    # 1. Build Matrices and Derivatives
    dRZZ = _drzz_matrix(p_rzz, dtype)
    
    derivs = [dRZZ]
        
    # Stack and contract
    grads = [
        hst_chain_rule(L, dL_dG, dU, n_sites, is_normalized)
        for dU in derivs
    ]
    return jnp.stack(grads)

def _update_circuit_from_hybrid_hw(circuit, params_tree_2q, meta_tree_2q, U_1q, idx_1q):
    """
    Update circuit with updated 2Q parameters and 1Q matrices.
    """
    dtype = circuit.dtype
    idx2mat_1q = {idx: U_1q[k] for k, idx in enumerate(idx_1q)}
    
    flat_i = 0
    for layer in sorted(circuit.layers, key=lambda l: l.layer_index):
        l_idx = layer.layer_index
        for g_idx, gate in enumerate(layer.iterate_gates(reverse=False)):
            if flat_i in idx2mat_1q:
                # 1Q Gate (Riemannian)
                gate.matrix = idx2mat_1q[flat_i]
            else:
                # 2Q Gate (Parameterized)
                if l_idx in params_tree_2q and g_idx in params_tree_2q[l_idx]:
                    theta = params_tree_2q[l_idx][g_idx]
                    meta = meta_tree_2q[l_idx][g_idx]
                    if meta["name"] == "RZZ":
                        # Reconstruct
                        p_rzz = theta[0]
                        gate.matrix = _rzz_matrix(p_rzz, dtype=dtype)
                        gate.params = _unflatten_vec_to_gate_params(theta, meta["splits"])
            flat_i += 1

def optimize_rzz(
        circ: Circuit, 
        mpo_ref: MPO, 
        *, 
        max_steps: int,
        max_bondim_env: int,
        svd_cutoff: float = 1e-12,
        lr: float = 1e-3,
        lr_1q: float | None = None,
        lr_2q: float | None = None,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        clip_grad_norm: float | None = None,
        bias_correction: bool = True,
        callback: Optional[Callable[..., bool]] = None,
        scheduler: Optional[object] = None,
        init_vertical_sweep = "top-down",
        use_ad: bool = False,
):
    """
    Hybrid optimizer: 1Q gates are updated Riemannian-ly on the unitary
    manifold; 2Q gates (RZZ) are updated as parameterized angles via Adam.
    """
    new_circ = circ.copy()

    # Decouple 1Q / 2Q base learning rates (fall back to the shared `lr`).
    # The 1Q (Riemannian) and 2Q (parameterized angle) updates live on
    # different geometries, so a single rate that suits one can be too large
    # for the other; allow them to be tuned independently.
    lr_1q_eff = lr if lr_1q is None else lr_1q
    lr_2q_eff = lr if lr_2q is None else lr_2q

    U1, U2, idx1, idx2 = build_buckets(new_circ)

    # Setup Riemannian Adam for 1Q gates
    opt_riem = RiemannianAdam(lr_1q_eff, betas, eps, clip_grad_norm, bias_correction)
    state_riem = opt_riem.init(U1)

    # Setup Parameterized Adam for 2Q gates
    params_tree_full, meta_tree_full = extract_params_tree(new_circ)
    params_tree_2q, meta_tree_2q = {}, {}
    for l_idx, gates in params_tree_full.items():
        params_tree_2q[l_idx], meta_tree_2q[l_idx] = {}, {}
        for g_idx, theta in gates.items():
            meta = meta_tree_full[l_idx][g_idx]
            if meta["arity"] == 2:
                params_tree_2q[l_idx][g_idx] = theta
                meta_tree_2q[l_idx][g_idx] = meta

    opt_2q = Adam(lr_2q_eff, betas, eps, clip_grad_norm, bias_correction)
    if use_ad:
        dtype = circ.dtype
        rzz_gen = lambda t: _rzz_matrix(t[0], dtype=dtype)
        opt_2q.register_param_grad("RZZ", opt_2q.make_ad_param_grad(rzz_gen))
    else:
        opt_2q.register_param_grad("RZZ", rzz_param_grad)

    u_2q = opt_2q.prepare_layout_from_trees(params_tree_2q, meta_tree_2q)

    history: List[float] = []
    for it in range(max_steps):
        overlap, grads_ordered, _ = overlap_and_environments(new_circ, mpo_ref, max_bondim_env=max_bondim_env, svd_cutoff=svd_cutoff, vertical_sweep=init_vertical_sweep)

        # Riemannian Step for 1Q gates
        if idx1:
            g1_env = jnp.stack([grads_ordered[i] for i in idx1], axis=0)
            c = 1.0 / (2 ** new_circ.n_sites) if mpo_ref.is_normalized else 1.0 / (2 ** (2 * new_circ.n_sites))
            scale = -2.0 * c * overlap
            U1, state_riem, _ = opt_riem.step(U1, scale * g1_env.conj(), state_riem)

        # Parameterized Step for 2Q gates
        if idx2:
            grads_2q = [grads_ordered[i] for i in idx2]
            u_2q, _, _ = opt_2q.step(
                u_2q,
                grads_2q,
                overlap,
                n_sites=new_circ.n_sites,
                is_normalized=mpo_ref.is_normalized,
            )

        # Sync back to circuit
        _update_circuit_from_hybrid_hw(new_circ, opt_2q.unflatten_to_params_tree(u_2q), meta_tree_2q, U1, idx1)

        loss = overlap_to_loss(overlap=overlap, kind="HST", n_sites=new_circ.n_sites, normalize=mpo_ref.is_normalized)
        history.append(loss)
        if it % 10 == 0: print(f"Step: {it}, Loss: {loss}")

        if scheduler:
            scheduler.step(loss, [opt_riem, opt_2q])

        if callback and callback(step=it, loss=loss):
            return new_circ, history

    return new_circ, history
