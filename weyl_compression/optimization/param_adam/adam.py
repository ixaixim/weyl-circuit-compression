from __future__ import annotations

import weyl_compression.jax_config

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Callable

import jax.numpy as jnp
import jax


ParamsTree = Dict[int, Dict[int, jnp.ndarray]]     # layer_index -> gate_index -> theta (1D)

# TODO: prepare layout from trees is useless. It collects the params in one vector, but In the end we update each gate individually anyways, because of different gate type...
#       remove the function prepare_layout_from_trees and just use the leafs.

# Small record for each gate in canonical order: (layer 0 L→R, then layer 1, ...)
@dataclass
class GateSlot:
    start: int                # start index in flat vector U
    stop: int                 # stop index (exclusive)
    name: str                 # gate.name
    meta: dict                # anything else you stored (qubits, splits, arity, ...)
    layer_idx: int
    gate_idx: int

class Adam: 

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        clip_grad_norm: Optional[float] = None,
        bias_correction: bool = True,
        wrap_angles: bool = False,
        ):

            self.lr = lr
            self.beta1, self.beta2 = betas
            self.eps = eps
            self.clip_grad_norm = clip_grad_norm
            self.bias_correction = bias_correction
            self.wrap_angles = wrap_angles

            self.step_count = 0
            self.m = None       # shape (G,)
            self.v = None       # shape (G,)
            self._slots: List[GateSlot] = []  # canonical order

            # registry: name -> fn(theta, dL_dG, overlap, meta, n_sites, is_normalized) -> grad_theta
            self.param_grad_fns: Dict[str, Callable[..., jnp.ndarray]] = {}

    # NOTE: in future versions, do not convert to flat vector, keep pytree
    # ---------------------------------------------------------------------
    # Call this once after you built params_tree/meta_tree to define layout
    # It also returns the initial flat parameter vector U0 (concatenation).
    def prepare_layout_from_trees(self, params_tree, meta_tree) -> jnp.ndarray:
        # canonical order = increasing layer_index, and within each layer gate_index 0..N-1
        layer_keys = sorted(params_tree.keys())
        leaves: List[jnp.ndarray] = []
        self._slots.clear()

        offset = 0
        for L in layer_keys:
            gate_keys = sorted(params_tree[L].keys())
            for gi in gate_keys:
                theta = params_tree[L][gi]               # 1D array (may be length 0)
                meta  = meta_tree[L][gi]
                p = int(theta.size)
                leaves.append(theta)
                self._slots.append(GateSlot(
                    start=offset, stop=offset+p, name=meta["name"], meta=meta,
                    layer_idx=L, gate_idx=gi
                ))
                offset += p

        U0 = jnp.concatenate([x.ravel() for x in leaves]) if leaves else jnp.zeros((0,), dtype=jnp.float64)
        G = U0.size
        self.step_count = 0
        self.m = jnp.zeros((G,), dtype=U0.dtype)
        self.v = jnp.zeros((G,), dtype=U0.dtype)
        return U0
    
    def unflatten_to_params_tree(self, U: jnp.ndarray) -> ParamsTree:
        """
        Reconstruct a ParamsTree (layer_idx -> gate_idx -> theta 1D) from flat vector U,
        using the canonical ordering encoded in self._slots.
        """
        params_tree: ParamsTree = {}
        # pre-create dicts to preserve keys if you like (optional)
        for slot in self._slots:
            if slot.layer_idx not in params_tree:
                params_tree[slot.layer_idx] = {}

        for slot in self._slots:
            s, e = slot.start, slot.stop
            # slice is length-0 for non-parameterized gates; that's fine
            theta = U[s:e]
            params_tree[slot.layer_idx][slot.gate_idx] = theta

        return params_tree


    # Register the chain-rule mapping for a given gate name.
    # fn signature: fn(theta: (p,), dL_dG, overlap, meta, n_sites, is_normalized) -> grad_theta: (p,)
    def register_param_grad(self, gate_name: str, fn: Callable[..., jnp.ndarray]):
        self.param_grad_fns[gate_name] = fn

    @staticmethod
    def make_ad_param_grad(gate_fn: Callable[[jnp.ndarray], jnp.ndarray]):
        """
        Creates a param_grad function that uses JAX AD to differentiate the gate_fn.
        gate_fn: theta -> U (unitary matrix)
        """
        def ad_grad_fn(theta, dL_dG, overlap, meta, n_sites, is_normalized):
            # Define local loss function: L = 1 - |Tr(E^dag U)|^2 / Norm
            # dL_dG is the environment E.
            def loss(t):
                U = gate_fn(t)
                # Ensure U has the same shape as dL_dG for element-wise multiplication
                U = U.reshape(dL_dG.shape)
                # Overlap O = Tr(E^dag U) = sum(E.conj * U)
                # NOTE: dL_dG is the environment tensor Env = d(Overlap)/dU. 
                # So Overlap = sum(Env * U). 
                ov = jnp.sum(dL_dG * U)
                mag_sq = jnp.abs(ov)**2
                
                if is_normalized:
                    denom = 2.0**n_sites
                else:
                    denom = 2.0**(2*n_sites)
                
                return 1.0 - mag_sq / denom
            
            # Compute gradient w.r.t theta
            return jax.grad(loss)(theta)
        return ad_grad_fn
          
    # ---------------------------------------------------------------------
    # Main update step operating on the flat parameter vector U
    def step(
        self,
        U: jnp.ndarray,                  # (G,) current flat params
        grad_euclid: List[jnp.ndarray],  # per-gate dL/dG in canonical order
        overlap: jnp.ndarray,
        *,
        n_sites: int,
        is_normalized: bool,
    ) -> Tuple[jnp.ndarray, dict, dict]:
        """
        Returns:
            U_next: (G,) updated parameters
            new_state: {'t': int, 'm': (G,), 'v': (G,)}
            stats: {'global_grad_norm': scalar, 'grad_norm': (G,), 'v': (G,), 'lr_eff': (G,)}
        """
        if len(grad_euclid) != len(self._slots):
            raise ValueError(f"grad_euclid length {len(grad_euclid)} != number of gates {len(self._slots)}")

        # --- 1) chain rule: per-gate dL/dG -> per-parameter gradient g (flat) ---
        g = jnp.zeros_like(U)
        for k, slot in enumerate(self._slots):
            s, e = slot.start, slot.stop
            if e == s:           # gate with no parameters
                continue
            theta = U[s:e]
            dL_dG = grad_euclid[k]
            try:
                fn = self.param_grad_fns[slot.name]
            except KeyError:
                raise KeyError(
                    f"No param-grad function registered for gate name '{slot.name}'. "
                    "Use Adam.register_param_grad(name, fn)."
                )
            g_gate = fn(theta, dL_dG, overlap, slot.meta, n_sites, is_normalized)       # shape (e-s,)

            # ensure shapes match
            if g_gate.shape != (e - s,):
                raise ValueError(f"param-grad for gate '{slot.name}' has shape {g_gate.shape}, expected {(e-s,)}")
            g = g.at[s:e].set(g_gate)

        # --- 2) optional global gradient clipping ---
        if self.clip_grad_norm is not None:
            # complex-safe L2 norm: ||g|| = sqrt(sum |g|^2)
            g_norm = jnp.sqrt(jnp.sum(jnp.abs(g) ** 2))
            scale = jnp.minimum(1.0, self.clip_grad_norm / (g_norm + 1e-16))
            g = g * scale
        else:
            g_norm = jnp.sqrt(jnp.sum(jnp.abs(g) ** 2))

        # --- 3) update biased moments (element-wise) ---
        t = self.step_count + 1
        m = self.beta1 * self.m + (1.0 - self.beta1) * g
        v = self.beta2 * self.v + (1.0 - self.beta2) * (jnp.abs(g) ** 2)  # works for real/complex

        # --- 4) bias correction (optional) ---
        if self.bias_correction:
            m_hat = m / (1.0 - self.beta1**t)
            v_hat = v / (1.0 - self.beta2**t)
        else:
            m_hat, v_hat = m, v

        # --- 5) parameter update ---
        denom = jnp.sqrt(v_hat) + self.eps
        step_vec = m_hat / denom
        U_next = U - self.lr * step_vec

        # optional: wrap angles into (-pi, pi]
        if self.wrap_angles:
            U_next = (U_next + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        # --- 6) commit state ---
        self.step_count = t
        self.m = m
        self.v = v

        # --- 7) stats (element-wise + global) ---
        stats = {
            "global_grad_norm": g_norm,                 # scalar
            "grad_norm": jnp.abs(g),                   # (G,)
            "v": v,                                    # (G,)
            "lr_eff": self.lr / denom,                 # (G,)
        }
        new_state = {"t": t, "m": m, "v": v}
        return U_next, new_state, stats
