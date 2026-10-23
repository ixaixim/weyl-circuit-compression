from setuptools import find_packages, setup

setup(
    name="weyl-circuit-compression",
    version="1.0.0",
    author="Neel Misciasci",
    author_email="neel.misciasci@tum.de",
    description="Tensor-network compression of Trotterized time-evolution circuits: "
                "Riemannian vs. hardware-aware parameterized optimization",
    packages=find_packages(include=["weyl_compression", "weyl_compression.*"]),
    python_requires=">=3.11",
    license="MIT",
    install_requires=[
        "numpy",
        "jax",
        "jaxlib",
        "qiskit",
        "PyYAML",
    ],
)
