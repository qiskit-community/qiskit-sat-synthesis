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

"""
Examples for synthesis of linear circuits with qudits.

As Qiskit does not support such circuits, the output is simply printed to the screen.
"""

import numpy as np

from qiskit.transpiler import CouplingMap

from qiskit_sat_synthesis.synthesize_linear_qudits import synthesize_linear_qudits_depth


def example_qutrits_lnn():
    coupling_map_line_3 = list(CouplingMap.from_full(3).get_edges())

    print("")
    print("===> Running example_qutrits_lnn")
    print("")
    mat = np.array([[1, 1, 2], [0, 0, 1], [0, 2, 1]])
    result = synthesize_linear_qudits_depth(
        mat=mat, qd=3, coupling_map=coupling_map_line_3, verbosity=1
    )
    print(result)
    print("")


def example_qutrits_full():
    coupling_map_full_3 = list(CouplingMap.from_full(3).get_edges())

    print("")
    print("===> Running example_qutrits_full")
    print("")
    mat = np.array([[1, 1, 2], [0, 0, 1], [0, 2, 1]])
    result = synthesize_linear_qudits_depth(
        mat=mat, qd=3, coupling_map=coupling_map_full_3, verbosity=1
    )
    print(result)
    print("")


if __name__ == "__main__":
    example_qutrits_lnn()
    example_qutrits_full()
