import pytest
import numpy as np
import jax.numpy as jnp
from weyl_compression.mpo.mpo_builder import create_dummy_mpo
from weyl_compression.mpo.mpo_dataclass import MPO

def is_left_canonical_tensor(T, atol=1e-6):
    """
    Checks if a tensor T(l, p_out, p_in, r) is left-canonical.
    Condition: sum_{l, p_out, p_in} T^*_{l, p_out, p_in, r} T_{l, p_out, p_in, r'} = delta_{r, r'}
    """
    r = T.shape[-1]
    T_mat = T.reshape(-1, r)
    T_mat_adj = T_mat.conj().T
    res = T_mat_adj @ T_mat

    return jnp.allclose(res, jnp.eye(r), atol=atol)

def is_right_canonical_tensor(T, atol=1e-6):
    """
    Checks if a tensor T(l, p_out, p_in, r) is right-canonical.
    Condition: sum_{p_out, p_in, r} T_{l, p_out, p_in, r} T^*_{k, p_out, p_in, r} = delta_{l, k}
    """
    l = T.shape[0]
    T_mat = T.reshape(l, -1)
    res = T_mat @ T_mat.conj().T

    return jnp.allclose(res, jnp.eye(l), atol=atol)

@pytest.mark.parametrize("n_sites", [4, 8])
def test_mpo_canonicity_flags_and_numerical_property(n_sites):
    # Create a random MPO with non-trivial bond dimensions
    bond_dims = [2, 4, 2, 1] if n_sites == 4 else [2, 4, 8, 4, 2, 2, 2, 1]
    mpo = create_dummy_mpo(bond_dims, random=True, seed=123)
    
    # --- Test Left Canonicalization ---
    mpo_l = mpo.copy()
    mpo_l.left_canonicalize(normalize=True)
    
    assert mpo_l.is_left_canonical
    assert not mpo_l.is_right_canonical
    
    for i in range(n_sites):
        assert is_left_canonical_tensor(mpo_l.tensors[i]), f"Tensor {i} failed left-canonicity check"
        
    # Test Dagger of Left-Canonical MPO
    mpo_l_dag = mpo_l.dagger()
    # In the current architecture (no bond swap), left-canonicity is preserved
    assert mpo_l_dag.is_left_canonical
    for i in range(n_sites):
        assert is_left_canonical_tensor(mpo_l_dag.tensors[i]), f"Daggered tensor {i} failed left-canonicity check"

    # --- Test Right Canonicalization ---
    mpo_r = mpo.copy()
    mpo_r.right_canonicalize(normalize=True)
    
    assert mpo_r.is_right_canonical
    assert not mpo_r.is_left_canonical
    
    for i in range(n_sites):
        assert is_right_canonical_tensor(mpo_r.tensors[i]), f"Tensor {i} failed right-canonicity check"
        
    # Test Dagger of Right-Canonical MPO
    mpo_r_dag = mpo_r.dagger()
    assert mpo_r_dag.is_right_canonical
    for i in range(n_sites):
        assert is_right_canonical_tensor(mpo_r_dag.tensors[i]), f"Daggered tensor {i} failed right-canonicity check"

def test_dagger_involution():
    """Checks that (M^dagger)^dagger == M."""
    mpo = create_dummy_mpo([2, 2, 1], random=True, seed=7)
    mpo_dag_dag = mpo.dagger().dagger()
    
    for t_orig, t_new in zip(mpo.tensors, mpo_dag_dag.tensors):
        np.testing.assert_allclose(t_orig, t_new, atol=1e-12)
