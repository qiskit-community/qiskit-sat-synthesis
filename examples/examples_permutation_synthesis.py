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

from qiskit.transpiler import CouplingMap
from qiskit_sat_synthesis.utils import perm_matrix
from qiskit_sat_synthesis.synthesize_permutation import synthesize_permutation_depth


def example_reversal():
    """Reversal permutation on a line (can be implemented in depth 6)."""
    print("")
    print("===> Running example_reversal")
    print("")

    coupling_map = list(CouplingMap.from_line(6).get_edges())
    perm = [5, 4, 3, 2, 1, 0]
    mat = perm_matrix(perm)

    result = synthesize_permutation_depth(
        mat=mat,
        coupling_map=coupling_map,
        print_solutions=True,
        check_solutions=True,
        verbosity=1,
    )
    print(result)
    print(result.circuit)
    print("")


if __name__ == "__main__":
    example_reversal()
