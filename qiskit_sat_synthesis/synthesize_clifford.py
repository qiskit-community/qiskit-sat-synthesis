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

"""Clifford synthesis."""

import numpy as np
from functools import partial

from qiskit.quantum_info import Clifford

from .sat_problem_clifford import SatProblemClifford
from .synthesize import SynthesisResult, synthesize_optimal
from .utils import make_downward


def create_depth2q_problem(
    target_clifford: Clifford,
    depth2q,
    coupling_map=None,
    coupling_map_list=None,
    full_2q=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    state_preparation_mode=False,
    max_2q_per_layer=None,
    max_1q_per_layer=None,
    max_num_2q_gates=None,
    max_num_1q_gates=None,
    max_num_unique_layers=None,
    restrict_search_space=False,
    optimize_2q_gates=False,
    optimize_1q_gates=False,
    verbosity=1,
    check_solutions=False,
    print_solutions=False,
    timeout_per_call=None,
    max_conflicts_per_call=None,
) -> SatProblemClifford:
    """
    Finds depth2q-optimal clifford based on Bravyi's approach.
    """
    if coupling_map is None and coupling_map_list is None:
        assert False, "Either coupling_map or coupling_map_list should be specified"
    if coupling_map is not None and coupling_map_list is not None:
        assert False, "Coupling_map and coupling_map_list cannot be both specified"

    if coupling_map is not None:
        coupling_maps = [make_downward(coupling_map)]
    else:
        coupling_maps = [
            make_downward(coupling_map) for coupling_map in coupling_map_list
        ]

    nq = target_clifford.num_qubits

    sat_problem = SatProblemClifford(nq, verbosity=verbosity)
    sat_problem.set_init_matrix_to_identity(nq)
    sat_problem.set_allow_layout_permutation(allow_layout_permutation)
    sat_problem.set_allow_final_permutation(allow_final_permutation)
    sat_problem.add_layer(gates=["S", "H", "SH", "HS", "SHS"], coupling_maps=[])

    all_2q_layer_ids = []

    for _ in range(depth2q):
        new_2q_layer_id = sat_problem.add_layer(
            gates=[
                "CX",
            ],
            coupling_maps=coupling_maps,
            full_2q=full_2q,
            max_num_2q_gates=max_2q_per_layer,
        )
        new_1q_layer_id = sat_problem.add_layer(
            gates=["SH", "HS"], coupling_maps=[], max_num_1q_gates=max_1q_per_layer
        )

        all_2q_layer_ids.append(new_2q_layer_id)
        sat_problem.add_1q_only_after_2q_constraint(new_2q_layer_id, new_1q_layer_id)
        sat_problem.add_nonempty_constraint(new_2q_layer_id)

    sat_problem.set_state_preparation_mode(state_preparation_mode)
    sat_problem.set_final_clifford(target_clifford)
    sat_problem.set_max_num_2q_gates(max_num_2q_gates)
    sat_problem.set_max_num_1q_gates(max_num_1q_gates)
    sat_problem.set_optimize_2q_gate(optimize_2q_gates)
    sat_problem.set_optimize_1q_gate(optimize_1q_gates)
    sat_problem.set_timeout_per_call(timeout_per_call)
    sat_problem.set_max_conflicts_per_call(max_conflicts_per_call)

    if max_num_unique_layers is not None:
        sat_problem.add_max_unique_layers_constraint(
            all_2q_layer_ids, max_num_unique_layers
        )

    if restrict_search_space:
        for i in range(1, len(all_2q_layer_ids)):
            # VALID TO ADD (for consecutive 2-qubit -- 1-qubit -- 2-qubit layers)
            layer1_id = all_2q_layer_ids[i - 1]
            layer2_id = layer1_id + 1
            layer3_id = layer2_id + 1
            sat_problem.add_cannot_push_2q_earlier_constraint(layer1_id, layer3_id)
            sat_problem.add_cannot_simplify_2q_1q_2q_constraint(
                layer1_id, layer2_id, layer3_id
            )
            sat_problem.add_commutation_2q_1q_2q_constraint(
                layer1_id, layer2_id, layer3_id
            )
    sat_problem.set_check_solutions(check_solutions)
    sat_problem.set_print_solutions(print_solutions)
    return sat_problem


def create_count2q_problem(
    target_clifford: Clifford,
    depth2q,
    coupling_map=None,
    coupling_map_list=None,
    full_2q=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    state_preparation_mode=False,
    max_2q_per_layer=None,
    max_1q_per_layer=None,
    max_num_2q_gates=None,
    max_num_1q_gates=None,
    max_num_unique_layers=None,
    restrict_search_space=False,
    optimize_2q_gates=False,
    optimize_1q_gates=False,
    verbosity=1,
    check_solutions=False,
    print_solutions=False,
    timeout_per_call=None,
    max_conflicts_per_call=None,
) -> SatProblemClifford:
    """
    Finds counth2q-optimal clifford based on Bravyi's approach.
    """
    if coupling_map is None and coupling_map_list is None:
        assert False, "Either coupling_map or coupling_map_list should be specified"
    if coupling_map is not None and coupling_map_list is not None:
        assert False, "Coupling_map and coupling_map_list cannot be both specified"

    if coupling_map is not None:
        coupling_maps = [make_downward(coupling_map)]
    else:
        coupling_maps = [
            make_downward(coupling_map) for coupling_map in coupling_map_list
        ]

    nq = target_clifford.num_qubits

    sat_problem = SatProblemClifford(nq, verbosity=verbosity)
    sat_problem.set_init_matrix_to_identity(nq)
    sat_problem.set_allow_layout_permutation(allow_layout_permutation)
    sat_problem.set_allow_final_permutation(allow_final_permutation)
    sat_problem.add_layer(gates=["S", "H", "SH", "HS", "SHS"], coupling_maps=[])

    all_2q_layer_ids = []

    for _ in range(depth2q):
        new_2q_layer_id = sat_problem.add_layer(
            gates=[
                "CX",
            ],
            coupling_maps=coupling_maps,
            full_2q=full_2q,
            max_num_2q_gates=1,
        )
        new_1q_layer_id = sat_problem.add_layer(
            gates=["SH", "HS"], coupling_maps=[], max_num_1q_gates=max_1q_per_layer
        )
        all_2q_layer_ids.append(new_2q_layer_id)
        sat_problem.add_1q_only_after_2q_constraint(new_2q_layer_id, new_1q_layer_id)
        sat_problem.add_nonempty_constraint(new_2q_layer_id)

    sat_problem.set_state_preparation_mode(state_preparation_mode)
    sat_problem.set_final_clifford(target_clifford)
    sat_problem.set_max_num_2q_gates(max_num_2q_gates)
    sat_problem.set_max_num_1q_gates(max_num_1q_gates)
    sat_problem.set_optimize_2q_gate(optimize_2q_gates)
    sat_problem.set_optimize_1q_gate(optimize_1q_gates)
    sat_problem.set_timeout_per_call(timeout_per_call)
    sat_problem.set_max_conflicts_per_call(max_conflicts_per_call)

    if max_num_unique_layers is not None:
        sat_problem.add_max_unique_layers_constraint(
            all_2q_layer_ids, max_num_unique_layers
        )

    if restrict_search_space:
        for i in range(1, len(all_2q_layer_ids)):
            layer1_id = all_2q_layer_ids[i - 1]
            layer2_id = layer1_id + 1
            layer3_id = layer2_id + 1
            sat_problem.add_layers_intersect_or_ordered(layer1_id, layer3_id)
            sat_problem.add_cannot_simplify_2q_1q_2q_constraint(
                layer1_id, layer2_id, layer3_id
            )
            sat_problem.add_commutation_2q_1q_2q_count_only_constraint(
                layer1_id, layer2_id, layer3_id
            )
    sat_problem.set_check_solutions(check_solutions)
    sat_problem.set_print_solutions(print_solutions)
    return sat_problem


def synthesize_clifford_depth(
    target_clifford,
    coupling_map=None,
    coupling_map_list=None,
    min_depth2q=0,
    max_depth2q=np.inf,
    full_2q=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    state_preparation_mode=False,
    max_2q_per_layer=None,
    max_1q_per_layer=None,
    max_num_2q_gates=None,
    max_num_1q_gates=None,
    max_num_unique_layers=None,
    optimize_2q_gates=False,
    optimize_1q_gates=False,
    check_solutions=False,
    print_solutions=False,
    max_solutions=1,
    restrict_search_space=True,
    timeout_per_call=None,
    max_conflicts_per_call=None,
    max_unsolved_depths=0,
    verbosity=1,
) -> SynthesisResult:
    sat_problem_fn = partial(
        create_depth2q_problem,
        coupling_map=coupling_map,
        coupling_map_list=coupling_map_list,
        full_2q=full_2q,
        allow_layout_permutation=allow_layout_permutation,
        allow_final_permutation=allow_final_permutation,
        state_preparation_mode=state_preparation_mode,
        max_2q_per_layer=max_2q_per_layer,
        max_1q_per_layer=max_1q_per_layer,
        max_num_2q_gates=max_num_2q_gates,
        max_num_1q_gates=max_num_1q_gates,
        max_num_unique_layers=max_num_unique_layers,
        restrict_search_space=restrict_search_space,
        optimize_1q_gates=optimize_1q_gates,
        optimize_2q_gates=optimize_2q_gates,
        verbosity=verbosity,
        check_solutions=check_solutions,
        print_solutions=print_solutions,
        timeout_per_call=timeout_per_call,
        max_conflicts_per_call=max_conflicts_per_call,
    )

    res = synthesize_optimal(
        target_obj=target_clifford,
        create_sat_problem_fn=sat_problem_fn,
        min_depth=min_depth2q,
        max_depth=max_depth2q,
        max_unsolved_depths=max_unsolved_depths,
        max_solutions=max_solutions,
        verbosity=verbosity,
    )

    return res


def synthesize_clifford_count(
    target_clifford,
    coupling_map=None,
    coupling_map_list=None,
    min_depth2q=0,
    max_depth2q=np.inf,
    full_2q=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    state_preparation_mode=False,
    max_2q_per_layer=None,
    max_1q_per_layer=None,
    max_num_2q_gates=None,
    max_num_1q_gates=None,
    max_num_unique_layers=None,
    optimize_2q_gates=False,
    optimize_1q_gates=False,
    check_solutions=False,
    print_solutions=False,
    max_solutions=1,
    restrict_search_space=True,
    timeout_per_call=None,
    max_conflicts_per_call=None,
    max_unsolved_depths=0,
    verbosity=1,
) -> SynthesisResult:
    sat_problem_fn = partial(
        create_count2q_problem,
        coupling_map=coupling_map,
        coupling_map_list=coupling_map_list,
        full_2q=full_2q,
        allow_layout_permutation=allow_layout_permutation,
        allow_final_permutation=allow_final_permutation,
        state_preparation_mode=state_preparation_mode,
        max_2q_per_layer=max_2q_per_layer,
        max_1q_per_layer=max_1q_per_layer,
        max_num_2q_gates=max_num_2q_gates,
        max_num_1q_gates=max_num_1q_gates,
        max_num_unique_layers=max_num_unique_layers,
        restrict_search_space=restrict_search_space,
        optimize_1q_gates=optimize_1q_gates,
        optimize_2q_gates=optimize_2q_gates,
        verbosity=verbosity,
        check_solutions=check_solutions,
        print_solutions=print_solutions,
        timeout_per_call=timeout_per_call,
        max_conflicts_per_call=max_conflicts_per_call,
    )

    res = synthesize_optimal(
        target_obj=target_clifford,
        create_sat_problem_fn=sat_problem_fn,
        min_depth=min_depth2q,
        max_depth=max_depth2q,
        max_unsolved_depths=max_unsolved_depths,
        max_solutions=max_solutions,
        verbosity=verbosity,
    )

    return res
