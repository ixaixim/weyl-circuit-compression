import weyl_compression.jax_config
import jax.numpy as jnp
from . import operators

class XYZModel:
    """
    Factory for representations of the 1D XYZ model Hamiltonian.

    H = Σᵢ(Jx·XᵢXᵢ₊₁ + Jy·YᵢYᵢ₊₁ + Jz·ZᵢZᵢ₊₁) + Σᵢ(hx·Xᵢ + hy·Yᵢ + hz·Zᵢ)

    Attributes:
        n_sites (int): Number of sites (spins) in the chain.
        Jx, Jy, Jz (float): Interaction strengths.
        hx, hy, hz (float): External magnetic field strengths.
        dtype (jnp.dtype): The data type for the tensors.
    """
    def __init__(
        self,
        n_sites: int,
        Jx: float,
        Jy: float,
        Jz: float,
        *,
        hx: float = 0.0,
        hy: float = 0.0,
        hz: float = 0.0,
        dtype: jnp.dtype = jnp.complex128
    ):
        if n_sites < 2:
            raise ValueError("Hamiltonian requires at least 2 sites.")
        self.n_sites = n_sites
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.dtype = dtype

    def _get_local_term_matrix(self, op: jnp.ndarray, site_idx: int) -> jnp.ndarray:
        """Pads a local operator to act on the full Hilbert space."""
        # Identity on the left
        H = jnp.eye(2**site_idx, dtype=self.dtype) if site_idx > 0 else 1.0
        # Operator on site i (and i+1 if two-site)
        H = jnp.kron(H, op)
        # Identity on the right
        # If op is 2x2 (1 qubit), bit_length is 2? No.
        # op.shape[0] is 2 or 4. 
        # If 2 (1 qubit), shift is 1. If 4 (2 qubits), shift is 2.
        op_sites = 1 if op.shape[0] == 2 else 2
        
        num_remaining_sites = self.n_sites - site_idx - op_sites
        if num_remaining_sites > 0:
            H = jnp.kron(H, jnp.eye(2**num_remaining_sites, dtype=self.dtype))
        
        return H

    def build_hamiltonian_matrix(self) -> jnp.ndarray:
        """
        Construct the full, dense Hamiltonian matrix for the XYZ chain.
        """
        ham_matrix = jnp.zeros((2**self.n_sites, 2**self.n_sites), dtype=self.dtype)

        # Interaction terms
        op_XX = jnp.kron(operators.X, operators.X)
        op_YY = jnp.kron(operators.Y, operators.Y)
        op_ZZ = jnp.kron(operators.Z, operators.Z)

        interaction_term = (
            self.Jx * op_XX + 
            self.Jy * op_YY + 
            self.Jz * op_ZZ
        )
        
        # Add interaction terms
        for i in range(self.n_sites - 1):
            ham_matrix += self._get_local_term_matrix(interaction_term, i)

        # Add field terms
        if self.hx != 0.0:
            for i in range(self.n_sites):
                ham_matrix += self._get_local_term_matrix(self.hx * operators.X, i)
        if self.hy != 0.0:
            for i in range(self.n_sites):
                ham_matrix += self._get_local_term_matrix(self.hy * operators.Y, i)
        if self.hz != 0.0:
            for i in range(self.n_sites):
                ham_matrix += self._get_local_term_matrix(self.hz * operators.Z, i)

        return ham_matrix
