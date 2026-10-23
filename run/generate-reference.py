"""
Generate the reference MPO U_ref = exp(-iHt) described by the `reference` block of a config:
a fine fourth-order Suzuki-Trotter circuit contracted into an MPO, optionally compressed to
the smallest bond dimension within `err_threshold`, then left-canonicalized and normalized.

Usage: python generate-reference.py <config.yml> [--out-dir DIR] [--force]
"""
import weyl_compression.jax_config

import argparse
from pathlib import Path

import jax.numpy as jnp

from weyl_compression.circuit.trotter.even_odd import even_odd_trotter_circuit
from weyl_compression.experiment import load_config, reference_filename
from weyl_compression.mpo.mpo_builder import circuit_to_mpo
from weyl_compression.mpo.mpo_dataclass import MPO
from weyl_compression.tensor_network.core_ops_helpers import hs_inner_product_from_mpo

RUN_DIR = Path(__file__).resolve().parent


def _compress_mpo(mpo: MPO, max_bondim: int) -> MPO:
    """
    Compress an MPO to max_bondim via a right-to-left SVD sweep.
    Starting from a left-canonical MPO, sweeps right-to-left, merging adjacent
    tensors and truncating the shared bond to max_bondim at each step.
    """
    tensors = [jnp.array(t) for t in mpo.tensors]
    n = len(tensors)

    for i in range(n - 1, 0, -1):
        T_l = tensors[i - 1]
        T_r = tensors[i]

        l, po, pi, _ = T_l.shape
        _, po2, pi2, r = T_r.shape

        theta = jnp.einsum('iabj, jcdk -> iabcdk', T_l, T_r)
        theta_mat = theta.reshape(l * po * pi, po2 * pi2 * r)

        U, S, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)

        k = min(max_bondim, len(S))
        tensors[i] = Vh[:k, :].reshape(k, po2, pi2, r)
        tensors[i - 1] = (U[:, :k] * S[None, :k]).reshape(l, po, pi, k)

    return MPO(tensors=tensors)


def _compute_mpo_error(mpo_ref: MPO, mpo_comp: MPO) -> float:
    """
    Compute the normalised Frobenius distance between two MPOs:
        err = 2 - 2 * Re(Tr(A†B)) / sqrt(Tr(A†A) * Tr(B†B))
    This equals ||A/||A|| - B/||B||||_F^2 and does not require explicit normalisation.
    """
    inner_12 = hs_inner_product_from_mpo(mpo_ref, mpo_comp)
    inner_11 = hs_inner_product_from_mpo(mpo_ref, mpo_ref)
    inner_22 = hs_inner_product_from_mpo(mpo_comp, mpo_comp)
    norm_product = jnp.sqrt(jnp.real(inner_11) * jnp.real(inner_22))
    return float(2.0 - 2.0 * jnp.real(inner_12) / norm_product)


def create_reference(cfg: dict) -> MPO:
    """
    Build the reference MPO of `cfg`.

    If `reference.compress` is set, decrement the bond dimension from `max_bond_dim` until
    the normalised Frobenius error relative to the full reference exceeds `err_threshold`,
    then keep the last MPO that stayed within the threshold.
    """
    h, ref = cfg["hamiltonian"], cfg["reference"]
    max_bondim_ref = ref["max_bond_dim"]

    target_circ = even_odd_trotter_circuit(
        n_sites=cfg["n_sites"],
        Jx=h["Jx"], Jy=h["Jy"], Jz=h["Jz"],
        hx=h["hx"], hy=h["hy"], hz=h["hz"],
        order=ref["order"],
        method='suzuki',
        dt=cfg["t"] / ref["reps"],
        reps=ref["reps"],
        dtype=jnp.complex128,
    )

    mpo_init = circuit_to_mpo(target_circ, max_bondim=max_bondim_ref, svd_cutoff=0.0)

    if ref["compress"]:
        bond_dim = max_bondim_ref
        err = 0.0
        while err < ref["err_threshold"]:
            bond_dim -= 1
            mpo_trial = _compress_mpo(mpo_init, bond_dim)
            err = _compute_mpo_error(mpo_init, mpo_trial)
            print(f"  bond_dim={bond_dim}: error={err:.6e}")

        bond_dim_final = bond_dim + 1
        if bond_dim_final < max_bondim_ref:
            print(f"Compressed reference to bond_dim={bond_dim_final} (threshold={ref['err_threshold']:.2e})")
            target_mpo = _compress_mpo(mpo_init, bond_dim_final)
        else:
            print(f"No compression possible: bond_dim={max_bondim_ref} already needed to meet "
                  f"threshold={ref['err_threshold']:.2e}")
            target_mpo = mpo_init
    else:
        target_mpo = mpo_init

    target_mpo.left_canonicalize(normalize=True)
    return target_mpo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: run/<model>/reference/)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing reference")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir or RUN_DIR / cfg["model"] / "reference"
    path = out_dir / reference_filename(cfg)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists; use --force to overwrite it.")

    mpo = create_reference(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpo.save_pickle(path)
    print(f"Saved reference to {path}")


if __name__ == "__main__":
    main()
