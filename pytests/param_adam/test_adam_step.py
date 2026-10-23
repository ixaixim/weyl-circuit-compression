import jax.numpy as jnp
import numpy as np
import pytest

import weyl_compression.jax_config  # ensure JAX dtype/config is initialized

from weyl_compression.optimization.param_adam.adam import Adam


def test_adam_step_single_gate_with_stub_grad():
    # Create a minimal single-parameter layout with one gate named 'RX'
    params_tree = {0: {0: jnp.array([1.5], dtype=jnp.float64)}}
    meta_tree = {0: {0: {"name": "RX"}}}

    # Instantiate Adam and prepare layout/state
    opt = Adam(lr=0.01, betas=(0.9, 0.999), eps=1e-8, clip_grad_norm=None, bias_correction=True)
    U0 = opt.prepare_layout_from_trees(params_tree, meta_tree)

    # Register a stub param-grad function that returns a fixed gradient g0, ignoring inputs
    g0 = jnp.array([0.3], dtype=U0.dtype)

    def stub_param_grad(theta, dL_dG, L, meta, n_sites, is_normalized):
        return g0

    opt.register_param_grad("RX", stub_param_grad)

    # Build dummy per-gate Euclidean gradient and overlap (ignored by stub)
    grads_ordered = [jnp.eye(2, dtype=jnp.complex128)]
    overlap = jnp.array(0.0, dtype=jnp.complex128)

    # Execute one optimizer step
    U1, new_state, stats = opt.step(
        U0,
        grads_ordered,
        overlap,
        n_sites=1,
        is_normalized=True,
    )

    # Expected values for t=1 with bias correction
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_expected = (1.0 - beta1) * g0
    v_expected = (1.0 - beta2) * (jnp.abs(g0) ** 2)

    m_hat = m_expected / (1.0 - beta1 ** 1)
    v_hat = v_expected / (1.0 - beta2 ** 1)
    denom = jnp.sqrt(v_hat) + eps
    step_vec = m_hat / denom
    U_expected = U0 - 0.01 * step_vec

    # Assertions on parameters and internal state
    np.testing.assert_allclose(np.asarray(U1), np.asarray(U_expected), rtol=1e-12, atol=1e-12)
    assert new_state["t"] == 1
    np.testing.assert_allclose(np.asarray(new_state["m"]), np.asarray(m_expected), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(new_state["v"]), np.asarray(v_expected), rtol=1e-12, atol=1e-12)

    # Stats sanity checks
    g_abs = np.abs(np.asarray(g0))[0]
    assert np.isclose(np.asarray(stats["global_grad_norm"]), g_abs)
    assert np.isclose(np.asarray(stats["grad_norm"])[0], g_abs)
    np.testing.assert_allclose(
        np.asarray(stats["lr_eff"])[0],
        0.01 / (g_abs + eps),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(np.asarray(stats["v"])[0], np.asarray(v_expected)[0], rtol=1e-12, atol=1e-12)

