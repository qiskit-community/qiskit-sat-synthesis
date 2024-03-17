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

"""Examples for Clifford synthesis."""

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap

from qiskit_sat_synthesis.synthesize_clifford import (
    synthesize_clifford_depth,
    synthesize_clifford_count,
)


def clifford_synthesis_lnn():
    """
    Simple Clifford synthesis tests for LNN connectivity.
    Varying options for count vs. depth, for layout_permutation and final_permutation.
    """

    qc = QuantumCircuit(4)
    qc.x(0)
    qc.s(1)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(1, 3)
    cliff = Clifford(qc)

    print("")
    print("===> Running clifford_synthesize_for_lnn")
    print("")

    coupling_map = list(CouplingMap.from_line(4).get_edges())
    for synthesis_fn in [synthesize_clifford_depth, synthesize_clifford_count]:
        for allow_layout_permutation in [False, True]:
            for allow_final_permutation in [False, True]:
                print(
                    f"=> Running with {synthesis_fn = }, {allow_final_permutation = }, {allow_layout_permutation = }"
                )
                result = synthesis_fn(
                    target_clifford=cliff,
                    coupling_map=coupling_map,
                    allow_final_permutation=allow_final_permutation,
                    allow_layout_permutation=allow_layout_permutation,
                    check_solutions=True,
                    verbosity=1,
                )
                print(result)
                print("")


if __name__ == "__main__":
    clifford_synthesis_lnn()
