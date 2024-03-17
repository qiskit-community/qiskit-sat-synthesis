# This code is part of Qiskit-Sat-Synthesis.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""QiskitSatSynthesis setup file."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name="qiskit-sat-synthesis",
    version="0.0.1",
    description="Sat-based synthesis plugins for Qiskit",
    license="Apache 2.0 License",
    author="Alexander Ivrii",
    author_email="alexi@il.ibm.com",
    packages=find_packages(),
    url="https://github.com/qiskit-community/qiskit-sat-synthesis",
    keywords="Synthesis, Quantum, Qiskit, Plugins, Sat",
    install_requires=REQUIREMENTS,
    entry_points={
        "qiskit.synthesis": [
            "permutation.sat_depth = qiskit_sat_synthesis.qiskit_synthesis_plugins:SatSynthesisPermutationDepth",
            "permutation.sat_count = qiskit_sat_synthesis.qiskit_synthesis_plugins:SatSynthesisPermutationCount",
            "linear_function.sat_depth = qiskit_sat_synthesis.qiskit_synthesis_plugins:SatSynthesisLinearFunctionDepth",
            "linear_function.sat_count = qiskit_sat_synthesis.qiskit_synthesis_plugins:SatSynthesisLinearFunctionCount",
            "clifford.sat_depth = qiskit_sat_synthesis.qiskit_synthesis_plugins:SatSynthesisCliffordDepth",
            "clifford.sat_count = qiskit_sat_synthesis.qiskit_synthesis_plugins:SatSynthesisCliffordCount",
        ]
    },
    python_requires=">=3.7",
)
