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

"""Examples for Linear Functions synthesis."""

import numpy as np

from qiskit.transpiler import CouplingMap

from qiskit_sat_synthesis.utils import extend_mat_with_ancillas
from qiskit_sat_synthesis.synthesize_linear import synthesize_linear_depth


def linear_mat_interesting():
    """This is an interesting matrix that shows how ancilla qubits can reduce depth of a linear circuit
    for full connectivity."""
    return np.array(
        [
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0],
        ]
    )


def linear_ancillas_reduce_depth():
    """This is an interesting example that shows how ancilla qubits can reduce depth of a linear circuit
    for full connectivity."""

    print("")
    print(f"===> Running linear_ancillas_reduce_depth")
    print("")

    mat = linear_mat_interesting()

    # This synthesizes depth-optimal circuit for the above 5x5 matrix using full connectivity on 5 qubits.
    coupling_map_full_5 = list(CouplingMap.from_full(5).get_edges())
    result = synthesize_linear_depth(
        mat,
        coupling_map=coupling_map_full_5,
        check_solutions=True,
        print_solutions=True,
        optimize_2q_gates=True,
        verbosity=1,
    )
    print(result)
    print("")

    # This synthesizes depth-optimal circuit for the above matrix (extended to 6 qubits)
    # using full connectivity on 6 qubits.
    mat_extended = extend_mat_with_ancillas(mat, [0, 1, 2, 3, 4], [5], [])

    coupling_map_full_6 = list(CouplingMap.from_full(6).get_edges())
    result = synthesize_linear_depth(
        mat_extended,
        coupling_map=coupling_map_full_6,
        check_solutions=True,
        print_solutions=True,
        optimize_2q_gates=True,
        verbosity=1,
    )
    print(result)
    print("")


if __name__ == "__main__":
    linear_ancillas_reduce_depth()
