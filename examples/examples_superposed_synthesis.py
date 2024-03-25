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

"""Examples for superposed states synthesis."""

from qiskit_sat_synthesis.synthesize_superposed import synthesize_superposed_depth


def simple_example_1():
    print("")
    print("===> Running simple_example_1")
    print("")

    target_state = [1] * 13
    coupling_map = [
        [11, 10],
        [10, 11],
        [11, 12],
        [12, 11],
        [10, 2],
        [2, 10],
        [12, 7],
        [7, 12],
        [2, 1],
        [1, 2],
        [2, 3],
        [3, 2],
        [7, 6],
        [6, 7],
        [7, 8],
        [8, 7],
        [1, 0],
        [0, 1],
        [3, 4],
        [4, 3],
        [6, 5],
        [5, 6],
        [8, 9],
        [9, 8],
    ]

    res = synthesize_superposed_depth(
        mat=target_state,
        coupling_map=coupling_map,
        print_solutions=True,
        check_solutions=True,
        optimize_2q_gates=True,
        root=None,
    )
    print(res)
    print("")


def example_unique_layers():
    color1 = [
        [0, 1],
        [2, 3],
        [7, 8],
        [9, 10],
        [11, 12],
        [17, 16],
        [19, 18],
        [1, 0],
        [3, 2],
        [8, 7],
        [10, 9],
        [12, 11],
        [16, 17],
        [18, 19],
    ]
    color2 = [
        [3, 4],
        [5, 7],
        [8, 9],
        [6, 11],
        [13, 12],
        [16, 14],
        [18, 17],
        [20, 15],
        [4, 3],
        [7, 5],
        [9, 8],
        [11, 6],
        [12, 13],
        [14, 16],
        [17, 18],
        [15, 20],
    ]
    color3 = [
        [0, 2],
        [4, 6],
        [1, 5],
        [15, 13],
        [20, 19],
        [14, 9],
        [10, 11],
        [2, 0],
        [6, 4],
        [5, 1],
        [13, 15],
        [19, 20],
        [9, 14],
        [11, 10],
    ]

    target_state = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    map_by_color = [color1, color2, color3]

    res = synthesize_superposed_depth(
        mat=target_state,
        coupling_map_list=map_by_color,
        print_solutions=True,
        check_solutions=True,
        optimize_2q_gates=False,
        max_num_unique_layers=3,
        full_2q=True,
        root=None,
    )
    print(res)
    print("")


if __name__ == "__main__":
    simple_example_1()
    example_unique_layers()
