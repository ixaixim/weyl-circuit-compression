from __future__ import annotations

import weyl_compression.jax_config
import jax.numpy as jnp

from dataclasses import dataclass
from typing import Optional, Tuple

from .geometry import transport_by_projection, retract_polar, project_to_tangent

def _fro_norm_sq(A):
    # supports (..., d, d)
    return jnp.real(jnp.sum(jnp.conj(A) * A, axis=(-2, -1)))

@dataclass
class RiemannianAdamState:
    step: int
    m: jnp.ndarray          # (..., d, d) tangent momentum attached to *current* U
    v: jnp.ndarray          # (...,) scalar 2nd moment per gate
    # Optional: last parameters if you prefer a different transport policy

class RiemannianAdam:
    """
    Riemannian Adam on the product of unitary groups (complex Stiefel, square case).

    - Projection: Π_U(X) = X - U sym(U^† X)
    - Retraction: polar retraction R_U(Ξ) = (U + Ξ)(I + Ξ^† Ξ)^{-1/2} or QR retraction
    - Transport:  T_{U→U'}(ζ) = Π_{U'}(ζ) with U' = R_U(Ξ_step)

    State policy:
      `state.m` is always a tangent vector at the *current* point U.
      After each step, we transport `m` to the new point U_next so the next
      iteration can update without an extra transport.

    Shapes:
      U: (G, d, d), grads: (G, d, d), m: (G, d, d), v: (G,)
      Scalar lr/betas/eps broadcast over G.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        clip_grad_norm: Optional[float] = None,
        bias_correction: bool = True,
    ):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.clip_grad_norm = clip_grad_norm
        self.bias_correction = bias_correction

    def init(self, U: jnp.ndarray) -> RiemannianAdamState:
        G, d, _ = U.shape
        m0 = jnp.zeros_like(U)           # tangent zero at U
        v0 = jnp.zeros((G,), dtype=U.dtype if jnp.iscomplexobj(U) else None)
        return RiemannianAdamState(step=0, m=m0, v=v0)

    def _maybe_clip(self, g: jnp.ndarray) -> jnp.ndarray:
        if self.clip_grad_norm is None:
            return g
        # global gate-wise clipping using Frobenius norm per gate
        norms = jnp.sqrt(_fro_norm_sq(g)) + 1e-16
        factor = jnp.minimum(1.0, self.clip_grad_norm / norms)
        # reshape for broadcasting over (d, d)
        while factor.ndim < g.ndim:
            factor = factor[..., None, None]
        return g * factor

    def step(
        self,
        U: jnp.ndarray,                 # (G, d, d) current unitary blocks
        grad_euclid: jnp.ndarray,       # (G, d, d) ambient/Euclidean gradient
        state: RiemannianAdamState,
    ) -> Tuple[jnp.ndarray, RiemannianAdamState, dict]:
        """
        One optimizer step. Returns (U_next, new_state, stats).
        """
        # 1) Riemannian gradient: project Euclidean grad to tangent at U  [Eq. (6),(8)]
        g_tan = project_to_tangent(U, grad_euclid)

        # Optional clipping in the Riemannian metric
        g_tan = self._maybe_clip(g_tan)

        # 2) First & second moments at the current point
        # state.m is already attached to U by construction (transported last step)
        m = self.beta1 * state.m + (1.0 - self.beta1) * g_tan

        # "v" tracks the scalar squared norm (Riemannian inner product) per gate
        g2 = _fro_norm_sq(g_tan)                    # <g,g>_U = ||g||_F^2 for the canonical metric
        v = self.beta2 * state.v + (1.0 - self.beta2) * g2

        # 3) Bias correction (per-gate scalars for v; matrices for m)
        t = state.step + 1
        if self.bias_correction:
            b1t = 1.0 - self.beta1**t
            b2t = 1.0 - self.beta2**t
            m_hat = m / b1t
            v_hat = v / b2t
        else:
            m_hat, v_hat = m, v

        # 4) Build the tangent step Ξ using scalar preconditioner (per gate)
        #    Ξ = - lr * m_hat / (sqrt(v_hat) + eps)   [ADAM-style]
        denom = jnp.sqrt(v_hat) + self.eps           # shape (G,)
        # broadcast denom over (d,d)
        while denom.ndim < m_hat.ndim:
            denom = denom[..., None, None]
        Xi = - self.lr * (m_hat / denom)

        # 5) Retract to the manifold to get the next parameters (use polar for consistency with transport)
        U_next = retract_polar(U, Xi)

        # 6) Transport momentum to the *new* tangent space for the next iteration  [Eq. (7)]
        m_next, _ = transport_by_projection(U, Xi, m)  # attach to U_next (same polar retraction)
        # Note: v is scalar; no transport required.

        new_state = RiemannianAdamState(step=t, m=m_next, v=v)

        stats = {
            "step": t,
            "grad_norm": jnp.sqrt(g2),    # (G,)
            "v": v,                      # (G,)
            "lr_eff": self.lr / (jnp.sqrt(v_hat) + self.eps),  # (G,)
        }
        return U_next, new_state, stats

