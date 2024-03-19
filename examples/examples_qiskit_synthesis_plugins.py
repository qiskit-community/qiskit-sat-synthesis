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

"""Examples of how to use qiskit transpiler high-level synthesis plugins."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import LinearFunction, PermutationGate
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig


def example_linear_function_plugin():
    # A 5x5 binary invertible matrix corresponding to a long-range CNOT-gate.
    mat = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ]

    # A quantum circuit which contains a linear function corresponding to our matrix.
    qc = QuantumCircuit(5)
    qc.append(LinearFunction(mat), [0, 1, 2, 3, 4])

    # The coupling map
    coupling_map = CouplingMap.from_line(5)

    # The high-level synthesis config to synthesize high-level objects in the circuit. Notably,it specifies
    # the "sat_depth" method to synthesize linear functions. The method accepts additional parameters:
    # the output verbosity and the option to minimize the number of 2-qubit gates once the minimum depth is
    # found.
    config = HLSConfig(
        linear_function=[("sat_depth", {"verbosity": 1, "optimize_2q_gates": True})]
    )

    # Running high-level synthesis transpiler pass and printing the transpiled circuit.
    qct = HighLevelSynthesis(
        hls_config=config, coupling_map=coupling_map, use_qubit_indices=True
    )(qc)
    print(qct)


def example_clifford_plugin():
    # A 5x5 binary invertible matrix corresponding to a long-range CNOT-gate.
    mat = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ]

    # A quantum circuit which contains a clifford corresponding to our matrix.
    qc = QuantumCircuit(5)
    qc.append(Clifford(LinearFunction(mat)), [0, 1, 2, 3, 4])

    # The coupling map
    coupling_map = CouplingMap.from_line(5)

    # The high-level synthesis config to synthesize high-level objects in the circuit. Notably,it specifies
    # the "sat_depth" method to synthesize cliffords. The method accepts additional parameters:
    # the output verbosity and the option to minimize the number of 2-qubit gates once the minimum depth is
    # found.
    config = HLSConfig(
        clifford=[("sat_depth", {"verbosity": 1, "optimize_2q_gates": True})]
    )

    # Running high-level synthesis transpiler pass and printing the transpiled circuit.
    qct = HighLevelSynthesis(
        hls_config=config, coupling_map=coupling_map, use_qubit_indices=True
    )(qc)
    print(qct)


def example_permutation_gate_plugin():
    # A quantum circuit which contains a reversal permutation.
    qc = QuantumCircuit(5)
    qc.append(PermutationGate([4, 3, 2, 1, 0]), [0, 1, 2, 3, 4])

    # The coupling map
    coupling_map = CouplingMap.from_line(5)

    # The high-level synthesis config to synthesize high-level objects in the circuit. Notably,it specifies
    # the "sat_depth" method to synthesize permutations. The method accepts additional parameters:
    # the output verbosity and the option to minimize the number of 2-qubit gates once the minimum depth is
    # found.
    config = HLSConfig(
        permutation=[("sat_depth", {"verbosity": 1, "optimize_2q_gates": True})]
    )

    # Running high-level synthesis transpiler pass and printing the transpiled circuit.
    qct = HighLevelSynthesis(
        hls_config=config, coupling_map=coupling_map, use_qubit_indices=True
    )(qc)
    print(qct)


if __name__ == "__main__":
    example_linear_function_plugin()
    example_clifford_plugin()
    example_permutation_gate_plugin()
