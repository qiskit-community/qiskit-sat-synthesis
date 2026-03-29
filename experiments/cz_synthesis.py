# This code is part of Qiskit-Sat-Synthesis.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Examples for Clifford synthesis."""

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library import LinearFunction

from qiskit_sat_synthesis.synthesize_clifford import (
    synthesize_clifford_depth,
    synthesize_clifford_count,
)


def full_cz(nq):
    qc = QuantumCircuit(nq)
    for i in range(0, nq):
        for j in range(i + 1, nq):
            qc.cz(i, j)
    return qc


def cz_up_to_linear():
    """Experiment; CZ up to linear.
    Connectivity: ?
    """
    nq = 6
    qc = full_cz(nq)
    print(qc)
    cliff = Clifford(qc)

    coupling_map = list(CouplingMap.from_full(nq).get_edges())
    print(coupling_map)

    allow_final_permutation = True
    for allow_final_linear in [False, True]:
        print("\n\n")
        print(f"Running with {allow_final_linear = }")
        synthesis_fn = synthesize_clifford_depth

        result = synthesis_fn(
            target_clifford=cliff,
            coupling_map=coupling_map,
            allow_final_linear=allow_final_linear,
            check_solutions=True,
            verbosity=1,
        )

        print("========FINAL RESULT============")
        print(result)
        print(result.circuit_with_permutations)


def cx_up_to_linear():
    """Experiment; CX up to linear.
    Connectivity: ?
    """
    nq = 4
    mat = np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    lf = LinearFunction(mat)

    qc = QuantumCircuit(4)
    qc.append(lf, [0, 1, 2, 3])
    print(qc)

    cliff = Clifford(qc)

    coupling_map = list(CouplingMap.from_full(nq).get_edges())
    print(coupling_map)

    synthesis_fn = synthesize_clifford_depth
    allow_final_permutation = True
    allow_final_linear = True

    result = synthesis_fn(
        target_clifford=cliff,
        coupling_map=coupling_map,
        allow_final_permutation=allow_final_permutation,
        allow_final_linear=allow_final_linear,
        check_solutions=True,
        verbosity=1,
    )

    print("========FINAL RESULT============")
    print(result)
    print(result.circuit_with_permutations)


if __name__ == "__main__":
    cz_up_to_linear()
    # cx_up_to_linear()
