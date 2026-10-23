import weyl_compression.jax_config
import jax.numpy as jnp
from typing import Callable, Optional

from weyl_compression.circuit.circuit_dataclasses import Circuit
from weyl_compression.mpo.mpo_dataclass import MPO
from weyl_compression.optimization.gradient import overlap_and_environments
from weyl_compression.optimization.param_adam.adam import Adam
from weyl_compression.optimization.riemannian_adam.riemannian_adam import RiemannianAdam
from weyl_compression.utils.pytree import extract_params_tree, _unflatten_vec_to_gate_params
from weyl_compression.optimization.utils import overlap_to_loss
from weyl_compression.optimization.utils import hst_chain_rule
from weyl_compression.utils.batching import build_buckets

def _fsim_matrix(theta_xy, theta_z, dtype=jnp.complex128):
    c = jnp.cos(2 * theta_xy)
    s = jnp.sin(2 * theta_xy)
    phase_z_m = jnp.exp(-1j * theta_z)
    phase_z_p = jnp.exp(1j * theta_z)
    
    mat = jnp.zeros((4, 4), dtype=dtype)
    mat = mat.at[0, 0].set(phase_z_m)
    mat = mat.at[1, 1].set(phase_z_p * c)
    mat = mat.at[1, 2].set(-1j * phase_z_p * s)
    mat = mat.at[2, 1].set(-1j * phase_z_p * s)
    mat = mat.at[2, 2].set(phase_z_p * c)
    mat = mat.at[3, 3].set(phase_z_m)
    return mat

def _d_fsim_matrix(theta_xy, theta_z, dtype=jnp.complex128):
    U = _fsim_matrix(theta_xy, theta_z, dtype)
    iHxy = jnp.zeros((4, 4), dtype=dtype)
    iHxy = iHxy.at[1, 2].set(-2j)
    iHxy = iHxy.at[2, 1].set(-2j)
    iHz = jnp.diag(jnp.array([-1j, 1j, 1j, -1j], dtype=dtype))
    return [iHxy @ U, iHz @ U]

def fsim_param_grad(theta, dL_dG, L, meta, n_sites, is_normalized):
    """
    Gradient for an fSim gate.
    theta = [theta_xy, theta_z]
    """
    dtype = dL_dG.dtype
    derivs = _d_fsim_matrix(theta[0], theta[1], dtype)
    grads = [hst_chain_rule(L, dL_dG, dU, n_sites, is_normalized) for dU in derivs]
    return jnp.stack(grads)

def _update_circuit_from_hybrid(circuit, params_tree_2q, meta_tree_2q, U_1q, idx_1q):
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
                    if meta["name"] == "fSim":
                        gate.matrix = _fsim_matrix(theta[0], theta[1], dtype=dtype)
                        gate.params = _unflatten_vec_to_gate_params(theta, meta["splits"])
            flat_i += 1

def optimize_fsim(
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
):
    """
    Hybrid optimizer for Heisenberg problem:
    - 2-qubit gates (fSim): Parameterized Adam on (theta_xy, theta_z).
    - 1-qubit gates: Riemannian Adam on SU(2) / U(2) unitary manifold.
    """
    new_circ = circ.copy()

    # Decouple 1Q / 2Q base learning rates (fall back to the shared `lr`).
    # The 1Q (Riemannian) and 2Q (parameterized angle) updates live on
    # different geometries, so a single rate that suits one can be too large
    # for the other; allow them to be tuned independently.
    lr_1q = lr if lr_1q is None else lr_1q
    lr_2q = lr if lr_2q is None else lr_2q

    # 1. Buckets for 1Q and 2Q
    U1, U2, idx1, idx2 = build_buckets(new_circ)

    # Adam/Riemannian-Adam momentum is initialized fresh on every call, so
    # chunking a training run into repeated optimize_fsim() calls to checkpoint
    # mid-trajectory resets momentum each time; use `callback` instead.
    # 2. Setup Riemannian Adam for 1Q gates
    opt_riem = RiemannianAdam(lr_1q, betas, eps, clip_grad_norm, bias_correction)
    state_riem = opt_riem.init(U1)
    
    # 3. Setup Parameterized Adam for 2Q gates
    params_tree_full, meta_tree_full = extract_params_tree(new_circ)
    params_tree_2q, meta_tree_2q = {}, {}
    for l_idx, gates in params_tree_full.items():
        params_tree_2q[l_idx], meta_tree_2q[l_idx] = {}, {}
        for g_idx, theta in gates.items():
            meta = meta_tree_full[l_idx][g_idx]
            if meta["arity"] == 2:
                params_tree_2q[l_idx][g_idx] = theta
                meta_tree_2q[l_idx][g_idx] = meta
                
    opt_2q = Adam(lr_2q, betas, eps, clip_grad_norm, bias_correction)
    opt_2q.register_param_grad("fSim", fsim_param_grad)
        
    u_2q = opt_2q.prepare_layout_from_trees(params_tree_2q, meta_tree_2q)
    
    history = []
    for it in range(max_steps):
        # 4. Euclidean Gradients (Environment)
        vert_dir = init_vertical_sweep if it % 2 == 0 else ("bottom-up" if init_vertical_sweep == "top-down" else "top-down")
        overlap, grads_ordered, _ = overlap_and_environments(
            new_circ, mpo_ref, 
            max_bondim_env=max_bondim_env, 
            svd_cutoff=svd_cutoff, 
            vertical_sweep=vert_dir
        )
        
        # 5. Riemannian Step for 1Q gates
        if idx1:
            g1_env = jnp.stack([grads_ordered[i] for i in idx1], axis=0)
            c = 1.0 / (2 ** new_circ.n_sites) if mpo_ref.is_normalized else 1.0 / (2 ** (2 * new_circ.n_sites))
            scale = -2.0 * c * overlap
            U1, state_riem, _ = opt_riem.step(U1, scale * g1_env.conj(), state_riem)
        
        # 6. Parameterized Step for 2Q gates
        if idx2:
            grads_2q = [grads_ordered[i] for i in idx2]
            u_2q, _, _ = opt_2q.step(
                u_2q,
                grads_2q,
                overlap,
                n_sites=new_circ.n_sites,
                is_normalized=mpo_ref.is_normalized,
            )
        
        # 7. Sync back to circuit
        _update_circuit_from_hybrid(new_circ, opt_2q.unflatten_to_params_tree(u_2q), meta_tree_2q, U1, idx1)
        
        # 8. Bookkeeping
        loss = overlap_to_loss(overlap=overlap, kind="HST", n_sites=new_circ.n_sites, normalize=mpo_ref.is_normalized)
        history.append(loss)
        if it % 10 == 0: print(f"Step: {it}, Loss: {loss}")
        
        if scheduler: scheduler.step(loss, [opt_riem, opt_2q])

        if callback and callback(step=it, loss=loss, circ=new_circ, overlap=overlap, grads_ordered=grads_ordered):
            return new_circ, history

    return new_circ, history
