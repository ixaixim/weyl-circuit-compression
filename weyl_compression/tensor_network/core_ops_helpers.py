import weyl_compression.jax_config

import jax.numpy as jnp
from weyl_compression.mpo.mpo_dataclass import MPO

def hs_inner_product_from_mpo(A_mpo, B_mpo):
    """
    Compute Tr(A^dag B) where A_mpo, B_mpo are MPOs.
    Assumes matching bond dimensions at boundaries (usually 1).
    """
    # Conjugate transpose A (dagger) at the MPO level
    A_dag = A_mpo.dagger()

    # Left environment starts as (1,1)
    # L_env corresponds to contraction of left bonds.
    # shape (A_bond, B_bond)
    # Assuming the first tensor has left bond dim 1
    # Check if bond dims match roughly or if they are 1.
    # Usually for MPO, first left bond is 1.
    
    L_env = jnp.eye(1, dtype=A_mpo[0].dtype).reshape(1, 1)

    # Contract site by site.
    for i in range(len(A_dag)):
        # A_dag[i] shape: (l_a, p_out, p_in, r_a)
        # B_mpo[i] shape: (l_b, p_out, p_in, r_b)
        # L_env shape: (l_a, l_b)
        # We want to contract L_env with left legs of A_dag and B_mpo
        # and contract physical legs of A_dag and B_mpo
        # Resulting shape: (r_a, r_b)
        
        # A_dag indices: a (left), b (p_in), c (p_out), d (right)
        # B_mpo indices: e (left), c (p_out), b (p_in), f (right)
        # L_env indices: a, e
        
        # We want to contract:
        # A_dag.p_in (b) with B_mpo.p_in (b)
        # A_dag.p_out (c) with B_mpo.p_out (c)
        L_env = jnp.einsum('ae, abcd, ecbf -> df', L_env, A_dag[i], B_mpo[i], optimize=True)

    # scalar overlap
    return L_env[0, 0]
