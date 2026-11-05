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
Synthesis of linear circuits with qudits.

As Qiskit does not support such circuits, the output is simply printed to the screen.
"""

from __future__ import annotations

import numpy as np
from functools import partial

from .synthesize import SynthesisResult, synthesize_optimal
from .utils import CouplingMapLike
from .sat_problem_linear_qudits import SatProblemLinearQudits


def create_depth2q_problem(
    depth2q: int,
    mat: np.ndarray,
    qd: int,
    coupling_map: CouplingMapLike | None = None,
    coupling_map_list: list[CouplingMapLike] | None = None,
    full_2q: bool = False,
    allow_layout_permutation: bool = False,
    allow_final_permutation: bool = False,
    max_2q_per_layer: int | None = None,
    max_num_2q_gates: int | None = None,
    max_num_unique_layers: int | None = None,
    restrict_search_space: bool = False,
    optimize_2q_gates: bool = False,
    verbosity: int = 1,
    check_solutions: bool = False,
    print_solutions: bool = False,
    timeout_per_call: int | None = None,
    max_conflicts_per_call: int | None = None,
) -> SatProblemLinearQudits:
    """
    Finds depth2q-optimal.
    """
    if coupling_map is None and coupling_map_list is None:
        assert False, "Either coupling_map or coupling_map_list should be specified"
    if coupling_map is not None and coupling_map_list is not None:
        assert False, "Coupling_map and coupling_map_list cannot be both specified"

    if coupling_map is not None:
        coupling_maps = [coupling_map]
    else:
        coupling_maps = coupling_map_list

    nq = len(mat)

    sat_problem = SatProblemLinearQudits(nq, qd, verbosity=verbosity)
    sat_problem.set_allow_layout_permutation(allow_layout_permutation)
    sat_problem.set_allow_final_permutation(allow_final_permutation)

    gates = []
    for c in range(1, qd):
        gates.append(tuple(["CX", c]))

    for _ in range(depth2q):
        new_2q_layer_id = sat_problem.add_layer(
            gates=gates,
            coupling_maps=coupling_maps,
            full_2q=full_2q,
            max_num_2q_gates=max_2q_per_layer,
        )
        sat_problem.add_nonempty_constraint(new_2q_layer_id)

    sat_problem.set_final_matrix(mat)
    sat_problem.set_max_num_2q_gates(max_num_2q_gates)
    sat_problem.set_optimize_2q_gate(optimize_2q_gates)
    sat_problem.set_timeout_per_call(timeout_per_call)
    sat_problem.set_max_conflicts_per_call(max_conflicts_per_call)

    sat_problem.set_check_solutions(check_solutions)
    sat_problem.set_print_solutions(print_solutions)
    return sat_problem


def synthesize_linear_qudits_depth(
    mat: np.ndarray,
    qd: int,
    coupling_map: CouplingMapLike | None = None,
    coupling_map_list: list[CouplingMapLike] | None = None,
    min_depth2q: int = 0,
    max_depth2q: int | float = np.inf,
    full_2q: bool = False,
    allow_layout_permutation: bool = False,
    allow_final_permutation: bool = False,
    max_2q_per_layer: int | None = None,
    max_num_2q_gates: int | None = None,
    max_num_unique_layers: int | None = None,
    optimize_2q_gates: bool = False,
    check_solutions: bool = False,
    print_solutions: bool = False,
    max_solutions: int = 1,
    restrict_search_space: bool = True,
    timeout_per_call: int | None = None,
    max_conflicts_per_call: int | None = None,
    max_unsolved_depths: int = 0,
    verbosity: int = 1,
) -> SynthesisResult:
    if verbosity >= 1:
        print(
            f"Running linear qudits synthesis with options: optimize depth2q, {qd = }, {allow_layout_permutation = }, {allow_final_permutation = }"
        )

    sat_problem_fn = partial(
        create_depth2q_problem,
        mat=mat,
        qd=qd,
        coupling_map=coupling_map,
        coupling_map_list=coupling_map_list,
        full_2q=full_2q,
        allow_layout_permutation=allow_layout_permutation,
        allow_final_permutation=allow_final_permutation,
        max_2q_per_layer=max_2q_per_layer,
        max_num_2q_gates=max_num_2q_gates,
        max_num_unique_layers=max_num_unique_layers,
        restrict_search_space=restrict_search_space,
        optimize_2q_gates=optimize_2q_gates,
        verbosity=verbosity,
        check_solutions=check_solutions,
        print_solutions=print_solutions,
        timeout_per_call=timeout_per_call,
        max_conflicts_per_call=max_conflicts_per_call,
    )

    res = synthesize_optimal(
        create_sat_problem_fn=sat_problem_fn,
        min_depth=min_depth2q,
        max_depth=max_depth2q,
        max_unsolved_depths=max_unsolved_depths,
        max_solutions=max_solutions,
        verbosity=verbosity,
    )

    return res
