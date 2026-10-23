# weyl_compression/hamiltonian/operators.py
import jax.numpy as jnp

# Define spin-1/2 operators (Pauli matrices)
# Using complex numbers for generality
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
