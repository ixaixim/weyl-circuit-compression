import weyl_compression.jax_config

import jax.numpy as jnp

def overlap_to_loss(
    overlap: jnp.ndarray,
    *,
    kind: str = "HST",
    n_sites: int = None,
    normalize: bool = True
) -> jnp.ndarray:
    """
    Turn a raw sum-of-traces (overlap) into a scalar loss.
    
    Parameters
    ----------
    overlap
        The real (or complex) scalar overlap returned by the gradient routine.
    kind : {'HST','frobenius'}
        Which cost function you’re using:
        - 'HST':      Squared HST loss:
            1 - |Tr(U_ref^† U_circ)|^2 / (||U_ref||_F^2 * ||U_circ||_F^2)

        - 'frobenius': loss ∝ ||U_ref - U_circ||_F^2
    n_sites
        Number of physical sites in the chain (for HST normalization).
    normalize
        Whether the MPO reference is normalized. Note: the circuit is always unitary (not normalized).
    """
    if kind.lower() == "hst":
        # maximal overlap for d-dim local Hilbert spaces is d**n_sites
        if (normalize is False):
            # MPO
            denom = 2.0 ** (2 * n_sites)
        else: 
            denom = 2.0 ** n_sites
        return 1.0 - (jnp.abs(overlap) ** 2) / denom
    
    elif kind.lower() == "frobenius":
        if normalize:
            denom = 2.0 ** (n_sites / 2.0)
        else:
            denom = 2.0 ** n_sites
        return 2.0 - 2.0 * jnp.real(overlap) / denom
    else:
        raise ValueError(f"Unknown loss kind: {kind}")


# z := Trace(U_ref^adj U_circ), i.e. complex holomorphic function (i.e. dz/dG*=0)
# note: the loss is real: this also means (df/dz)* = df/dz*.

# Let G = G(θ) with real θ. Then
#
#     dL/dθ = ∑_i ∑_j (∂L/∂G_ij) (dG_ij/dθ) + ∑_i ∑_j (∂L/∂Ḡ_ij) (dḠ_ij/dθ).
#
# Since L is holomorphic, we have ∂L/∂Ḡ = 0. Therefore, the differential reduces to
#
#     dL/dθ = ∑_i ∑_j (∂L/∂G_ij) (dG_ij/dθ) = Tr(∂L/∂G^T dG/dθ), which is the sum implemented as
# 

def hst_chain_rule(z, dz_dG, dG_dtheta, n_sites, is_normalized):
    d = 2.0**n_sites
    denom = d*d if not is_normalized else d
    # implements  sum_{ij} (∂f/∂G_ij) * (dG_ij/dθ)
    return -(2.0 / denom) * jnp.real(jnp.conjugate(z) * jnp.trace(dz_dG.T @ dG_dtheta))
