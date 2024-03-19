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

"""Linear Function synthesis."""

import numpy as np
from functools import partial

from .synthesize import SynthesisResult, synthesize_optimal
from .sat_problem_linear import SatProblemLinear
from .utils import extend_identity_with_ancillas, extend_mat_with_ancillas


# Standard problem for linear synthesis


def create_depth2q_problem(
    depth2q,
    mat,
    main_qubits=None,
    dirty_ancilla_qubits=None,
    clean_ancilla_qubits=None,
    coupling_map=None,
    coupling_map_list=None,
    full_2q=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    max_2q_per_layer=None,
    max_num_2q_gates=None,
    max_num_unique_layers=None,
    restrict_search_space=False,
    optimize_2q_gates=False,
    verbosity=1,
    check_solutions=False,
    print_solutions=False,
    timeout_per_call=None,
    max_conflicts_per_call=None,
) -> SatProblemLinear:
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

    if main_qubits is None:
        main_qubits = list(range(len(mat)))

    if dirty_ancilla_qubits is None:
        dirty_ancilla_qubits = []

    if clean_ancilla_qubits is None:
        clean_ancilla_qubits = []

    init_matrix = extend_identity_with_ancillas(
        main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
    )
    final_matrix = extend_mat_with_ancillas(
        mat, main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
    )

    nq = len(main_qubits) + len(dirty_ancilla_qubits) + len(clean_ancilla_qubits)

    sat_problem = SatProblemLinear(nq, verbosity=verbosity)
    sat_problem.set_init_matrix(init_matrix)

    sat_problem.set_allow_layout_permutation(allow_layout_permutation)
    sat_problem.set_allow_final_permutation(allow_final_permutation)

    all_2q_layer_ids = []

    for _ in range(depth2q):
        new_2q_layer_id = sat_problem.add_layer(
            gates=["CX"],
            coupling_maps=coupling_maps,
            full_2q=full_2q,
            max_num_2q_gates=max_2q_per_layer,
        )
        all_2q_layer_ids.append(new_2q_layer_id)
        # VALID TO ADD
        # - SatLayerNonEmptyConstraint for all layers
        sat_problem.add_nonempty_constraint(new_2q_layer_id)

    sat_problem.set_final_matrix(final_matrix)
    sat_problem.set_max_num_2q_gates(max_num_2q_gates)
    sat_problem.set_optimize_2q_gate(optimize_2q_gates)
    sat_problem.set_timeout_per_call(timeout_per_call)
    sat_problem.set_max_conflicts_per_call(max_conflicts_per_call)

    if max_num_unique_layers is not None:
        sat_problem.add_max_unique_layers_constraint(
            all_2q_layer_ids, max_num_unique_layers
        )

    if restrict_search_space:
        # VALID TO ADD (for consecutive 2-qubit layers)
        # - SatCannotPush2QEarlierConstraint
        # - SatCannotSimplify2Q2QConstraint
        # - SatCommutation2Q2QConstraint
        for i in range(1, len(all_2q_layer_ids)):
            prev_layer_id = all_2q_layer_ids[i - 1]
            next_layer_id = all_2q_layer_ids[i]
            sat_problem.add_cannot_push_2q_earlier_constraint(
                prev_layer_id, next_layer_id
            )
            sat_problem.add_cannot_simplify_2q_2q_constraint(
                prev_layer_id, next_layer_id
            )
            sat_problem.add_commutation_2q_2q_constraint(prev_layer_id, next_layer_id)

    sat_problem.set_check_solutions(check_solutions)
    sat_problem.set_print_solutions(print_solutions)
    return sat_problem


def create_count2q_problem(
    depth2q,
    mat,
    main_qubits=None,
    dirty_ancilla_qubits=None,
    clean_ancilla_qubits=None,
    coupling_map=None,
    coupling_map_list=None,
    full_2q=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    max_2q_per_layer=None,
    max_num_2q_gates=None,
    max_num_unique_layers=None,
    restrict_search_space=False,
    optimize_2q_gates=False,
    verbosity=1,
    check_solutions=False,
    print_solutions=False,
    timeout_per_call=None,
    max_conflicts_per_call=None,
) -> SatProblemLinear:
    """
    Finds counth2q-optimal.
    """
    if coupling_map is None and coupling_map_list is None:
        assert False, "Either coupling_map or coupling_map_list should be specified"
    if coupling_map is not None and coupling_map_list is not None:
        assert False, "Coupling_map and coupling_map_list cannot be both specified"

    if coupling_map is not None:
        coupling_maps = [coupling_map]
    else:
        coupling_maps = coupling_map_list

    if main_qubits is None:
        main_qubits = list(range(len(mat)))

    if dirty_ancilla_qubits is None:
        dirty_ancilla_qubits = []

    if clean_ancilla_qubits is None:
        clean_ancilla_qubits = []

    init_matrix = extend_identity_with_ancillas(
        main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
    )
    final_matrix = extend_mat_with_ancillas(
        mat, main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
    )

    nq = len(main_qubits) + len(dirty_ancilla_qubits) + len(clean_ancilla_qubits)

    sat_problem = SatProblemLinear(nq, verbosity=verbosity)

    sat_problem.set_init_matrix(init_matrix)
    sat_problem.set_allow_layout_permutation(allow_layout_permutation)
    sat_problem.set_allow_final_permutation(allow_final_permutation)

    all_2q_layer_ids = []

    for _ in range(depth2q):
        new_2q_layer_id = sat_problem.add_layer(
            gates=["CX"],
            coupling_maps=coupling_maps,
            full_2q=full_2q,
            max_num_2q_gates=1,
        )
        all_2q_layer_ids.append(new_2q_layer_id)
        # VALID TO ADD
        # - SatLayerNonEmptyConstraint for all layers
        sat_problem.add_nonempty_constraint(new_2q_layer_id)

    sat_problem.set_final_matrix(final_matrix)
    sat_problem.set_max_num_2q_gates(max_num_2q_gates)
    sat_problem.set_optimize_2q_gate(optimize_2q_gates)
    sat_problem.set_timeout_per_call(timeout_per_call)
    sat_problem.set_max_conflicts_per_call(max_conflicts_per_call)

    if max_num_unique_layers is not None:
        sat_problem.add_max_unique_layers_constraint(
            all_2q_layer_ids, max_num_unique_layers
        )

    if restrict_search_space:
        # VALID TO ADD (for consecutive 2-qubit layers)
        # - SatLayersIntersectOrOrderedConstraint
        # - SatCannotSimplify2Q2QConstraint
        # - SatCommutation2Q2QConstraint
        for i in range(1, len(all_2q_layer_ids)):
            prev_layer_id = all_2q_layer_ids[i - 1]
            next_layer_id = all_2q_layer_ids[i]
            sat_problem.add_layers_intersect_or_ordered(prev_layer_id, next_layer_id)
            sat_problem.add_cannot_simplify_2q_2q_constraint(
                prev_layer_id, next_layer_id
            )
            sat_problem.add_commutation_2q_2q_constraint(prev_layer_id, next_layer_id)

    sat_problem.set_check_solutions(check_solutions)
    sat_problem.set_print_solutions(print_solutions)
    return sat_problem


# Extended version by Simon Martial that allows mid-circuit measurements


def create_depth2q_problem_with_measurements(
    depth2q,
    mat,
    main_qubits=None,
    dirty_ancilla_qubits=None,
    clean_ancilla_qubits=None,
    coupling_map=None,
    coupling_map_list=None,
    full_2q=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    max_2q_per_layer=None,
    max_num_2q_gates=None,
    max_num_unique_layers=None,
    restrict_search_space=False,
    optimize_2q_gates=False,
    verbosity=1,
    check_solutions=False,
    print_solutions=False,
    timeout_per_call=None,
    max_conflicts_per_call=None,
) -> SatProblemLinear:
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

    if main_qubits is None:
        main_qubits = list(range(len(mat)))

    if dirty_ancilla_qubits is None:
        dirty_ancilla_qubits = []

    if clean_ancilla_qubits is None:
        clean_ancilla_qubits = []

    init_matrix = extend_identity_with_ancillas(
        main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
    )
    final_matrix = extend_mat_with_ancillas(
        mat, main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
    )

    nq = len(main_qubits) + len(dirty_ancilla_qubits) + len(clean_ancilla_qubits)

    sat_problem = SatProblemLinear(nq, verbosity=verbosity)
    sat_problem.set_init_matrix(init_matrix)

    sat_problem.set_allow_layout_permutation(allow_layout_permutation)
    sat_problem.set_allow_final_permutation(allow_final_permutation)

    all_2q_layer_ids = []

    for _ in range(depth2q):
        new_2q_layer_id = sat_problem.add_layer(
            gates=["CX", "MEASUREMENT"],
            coupling_maps=coupling_maps,
            full_2q=full_2q,
            max_num_2q_gates=max_2q_per_layer,
        )
        all_2q_layer_ids.append(new_2q_layer_id)
        # VALID TO ADD
        # - SatLayerNonEmptyConstraint for all layers
        # sat_problem.add_nonempty_constraint(new_2q_layer_id)

    sat_problem.set_final_matrix(final_matrix)
    sat_problem.set_max_num_2q_gates(max_num_2q_gates)
    sat_problem.set_optimize_2q_gate(optimize_2q_gates)
    sat_problem.set_timeout_per_call(timeout_per_call)
    sat_problem.set_max_conflicts_per_call(max_conflicts_per_call)

    if max_num_unique_layers is not None:
        sat_problem.add_max_unique_layers_constraint(
            all_2q_layer_ids, max_num_unique_layers
        )

    # ToDo: rethink these optimizations
    # if restrict_search_space:
    #     # VALID TO ADD (for consecutive 2-qubit layers)
    #     # - SatCannotPush2QEarlierConstraint
    #     # - SatCannotSimplify2Q2QConstraint
    #     # - SatCommutation2Q2QConstraint
    #     for i in range(1, len(all_2q_layer_ids)):
    #         prev_layer_id = all_2q_layer_ids[i - 1]
    #         next_layer_id = all_2q_layer_ids[i]
    #         sat_problem.add_cannot_push_2q_earlier_constraint(
    #             prev_layer_id, next_layer_id
    #         )
    #         sat_problem.add_cannot_simplify_2q_2q_constraint(
    #             prev_layer_id, next_layer_id
    #         )
    #         sat_problem.add_commutation_2q_2q_constraint(prev_layer_id, next_layer_id)

    sat_problem.set_check_solutions(check_solutions)
    sat_problem.set_print_solutions(print_solutions)
    return sat_problem


def create_count2q_problem_with_measurements(
    depth2q,
    mat,
    main_qubits=None,
    dirty_ancilla_qubits=None,
    clean_ancilla_qubits=None,
    coupling_map=None,
    coupling_map_list=None,
    full_2q=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    max_2q_per_layer=None,
    max_num_2q_gates=None,
    max_num_unique_layers=None,
    restrict_search_space=False,
    optimize_2q_gates=False,
    verbosity=1,
    check_solutions=False,
    print_solutions=False,
    timeout_per_call=None,
    max_conflicts_per_call=None,
) -> SatProblemLinear:
    """
    Finds counth2q-optimal.
    """
    if coupling_map is None and coupling_map_list is None:
        assert False, "Either coupling_map or coupling_map_list should be specified"
    if coupling_map is not None and coupling_map_list is not None:
        assert False, "Coupling_map and coupling_map_list cannot be both specified"

    if coupling_map is not None:
        coupling_maps = [coupling_map]
    else:
        coupling_maps = coupling_map_list

    if main_qubits is None:
        main_qubits = list(range(len(mat)))

    if dirty_ancilla_qubits is None:
        dirty_ancilla_qubits = []

    if clean_ancilla_qubits is None:
        clean_ancilla_qubits = []

    init_matrix = extend_identity_with_ancillas(
        main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
    )
    final_matrix = extend_mat_with_ancillas(
        mat, main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
    )

    nq = len(main_qubits) + len(dirty_ancilla_qubits) + len(clean_ancilla_qubits)

    sat_problem = SatProblemLinearWithMeasurements(nq, verbosity=verbosity)

    sat_problem.set_init_matrix(init_matrix)
    sat_problem.set_allow_layout_permutation(allow_layout_permutation)
    sat_problem.set_allow_final_permutation(allow_final_permutation)

    all_2q_layer_ids = []

    for _ in range(depth2q):
        new_2q_layer_id = sat_problem.add_layer(
            gates=["CX"],
            coupling_maps=coupling_maps,
            full_2q=full_2q,
            max_num_2q_gates=1,
        )
        all_2q_layer_ids.append(new_2q_layer_id)
        # VALID TO ADD
        # - SatLayerNonEmptyConstraint for all layers
        sat_problem.add_nonempty_constraint(new_2q_layer_id)

    sat_problem.set_final_matrix(final_matrix)
    sat_problem.set_max_num_2q_gates(max_num_2q_gates)
    sat_problem.set_optimize_2q_gate(optimize_2q_gates)
    sat_problem.set_timeout_per_call(timeout_per_call)
    sat_problem.set_max_conflicts_per_call(max_conflicts_per_call)

    if max_num_unique_layers is not None:
        sat_problem.add_max_unique_layers_constraint(
            all_2q_layer_ids, max_num_unique_layers
        )

    if restrict_search_space:
        # VALID TO ADD (for consecutive 2-qubit layers)
        # - SatLayersIntersectOrOrderedConstraint
        # - SatCannotSimplify2Q2QConstraint
        # - SatCommutation2Q2QConstraint
        for i in range(1, len(all_2q_layer_ids)):
            prev_layer_id = all_2q_layer_ids[i - 1]
            next_layer_id = all_2q_layer_ids[i]
            sat_problem.add_layers_intersect_or_ordered(prev_layer_id, next_layer_id)
            sat_problem.add_cannot_simplify_2q_2q_constraint(
                prev_layer_id, next_layer_id
            )
            sat_problem.add_commutation_2q_2q_constraint(prev_layer_id, next_layer_id)

    sat_problem.set_check_solutions(check_solutions)
    sat_problem.set_print_solutions(print_solutions)
    return sat_problem


# Synthesis utils


def synthesize_linear_depth(
    mat,
    main_qubits=None,
    dirty_ancilla_qubits=None,
    clean_ancilla_qubits=None,
    coupling_map=None,
    coupling_map_list=None,
    min_depth2q=0,
    max_depth2q=np.inf,
    full_2q=False,
    allow_measurements=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    max_2q_per_layer=None,
    max_num_2q_gates=None,
    max_num_unique_layers=None,
    optimize_2q_gates=False,
    check_solutions=False,
    print_solutions=False,
    max_solutions=1,
    restrict_search_space=True,
    timeout_per_call=None,
    max_conflicts_per_call=None,
    max_unsolved_depths=0,
    verbosity=1,
) -> SynthesisResult:
    if verbosity >= 1:
        print(
            f"Running linear synthesis with options: optimize depth2q, {allow_layout_permutation = }, {allow_final_permutation = }"
        )

    if not allow_measurements:
        sat_problem_fn = partial(
            create_depth2q_problem,
            mat=mat,
            main_qubits=main_qubits,
            dirty_ancilla_qubits=dirty_ancilla_qubits,
            clean_ancilla_qubits=clean_ancilla_qubits,
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
    else:
        sat_problem_fn = partial(
            create_depth2q_problem_with_measurements,
            mat=mat,
            main_qubits=main_qubits,
            dirty_ancilla_qubits=dirty_ancilla_qubits,
            clean_ancilla_qubits=clean_ancilla_qubits,
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


def synthesize_linear_count(
    mat,
    main_qubits=None,
    dirty_ancilla_qubits=None,
    clean_ancilla_qubits=None,
    coupling_map=None,
    coupling_map_list=None,
    min_depth2q=0,
    max_depth2q=np.inf,
    full_2q=False,
    allow_measurements=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    max_2q_per_layer=None,
    max_num_2q_gates=None,
    max_num_unique_layers=None,
    optimize_2q_gates=False,
    check_solutions=False,
    print_solutions=False,
    max_solutions=1,
    restrict_search_space=True,
    timeout_per_call=None,
    max_conflicts_per_call=None,
    max_unsolved_depths=0,
    verbosity=1,
) -> SynthesisResult:
    if verbosity >= 1:
        print(
            f"Running linear synthesis with options: optimize count2q, {allow_layout_permutation = }, {allow_final_permutation = }"
        )

    if not allow_measurements:
        sat_problem_fn = partial(
            create_count2q_problem,
            mat=mat,
            main_qubits=main_qubits,
            dirty_ancilla_qubits=dirty_ancilla_qubits,
            clean_ancilla_qubits=clean_ancilla_qubits,
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
    else:
        sat_problem_fn = partial(
            create_count2q_problem_with_measurements,
            mat=mat,
            main_qubits=main_qubits,
            dirty_ancilla_qubits=dirty_ancilla_qubits,
            clean_ancilla_qubits=clean_ancilla_qubits,
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


def synthesize_linear_count_depth(
    mat,
    main_qubits=None,
    dirty_ancilla_qubits=None,
    clean_ancilla_qubits=None,
    coupling_map=None,
    coupling_map_list=None,
    min_depth2q=0,
    max_depth2q=np.inf,
    full_2q=False,
    allow_measurements=False,
    allow_layout_permutation=False,
    allow_final_permutation=False,
    max_2q_per_layer=None,
    max_num_2q_gates=None,
    max_num_unique_layers=None,
    optimize_2q_gates=False,
    check_solutions=False,
    print_solutions=False,
    max_solutions=1,
    restrict_search_space=True,
    timeout_per_call=None,
    max_conflicts_per_call=None,
    max_unsolved_depths=0,
    verbosity=1,
) -> SynthesisResult:
    if verbosity >= 1:
        print(
            f"Running linear synthesis with options: optimize count2q--depth2q, {allow_layout_permutation = }, {allow_final_permutation = }"
        )
    res_count = synthesize_linear_count(
        mat,
        main_qubits=main_qubits,
        dirty_ancilla_qubits=dirty_ancilla_qubits,
        clean_ancilla_qubits=clean_ancilla_qubits,
        coupling_map=coupling_map,
        coupling_map_list=coupling_map_list,
        min_depth2q=min_depth2q,
        max_depth2q=max_depth2q,
        full_2q=full_2q,
        allow_measurements=allow_measurements,
        allow_layout_permutation=allow_layout_permutation,
        allow_final_permutation=allow_final_permutation,
        max_2q_per_layer=max_2q_per_layer,
        max_num_2q_gates=max_num_2q_gates,
        max_num_unique_layers=max_num_unique_layers,
        optimize_2q_gates=optimize_2q_gates,
        check_solutions=check_solutions,
        print_solutions=print_solutions,
        max_solutions=max_solutions,
        restrict_search_space=restrict_search_space,
        verbosity=verbosity,
        timeout_per_call=timeout_per_call,
        max_conflicts_per_call=max_conflicts_per_call,
        max_unsolved_depths=0,
    )
    res_depth = synthesize_linear_depth(
        mat,
        main_qubits=main_qubits,
        dirty_ancilla_qubits=dirty_ancilla_qubits,
        clean_ancilla_qubits=clean_ancilla_qubits,
        coupling_map=coupling_map,
        coupling_map_list=coupling_map_list,
        min_depth2q=min_depth2q,
        max_depth2q=max_depth2q,
        full_2q=full_2q,
        allow_measurements=allow_measurements,
        allow_layout_permutation=allow_layout_permutation,
        allow_final_permutation=allow_final_permutation,
        max_2q_per_layer=max_2q_per_layer,
        max_num_2q_gates=res_count.min_solution_depth,
        max_num_unique_layers=max_num_unique_layers,
        optimize_2q_gates=optimize_2q_gates,
        check_solutions=check_solutions,
        print_solutions=print_solutions,
        max_solutions=max_solutions,
        restrict_search_space=restrict_search_space,
        verbosity=verbosity,
        timeout_per_call=timeout_per_call,
        max_conflicts_per_call=max_conflicts_per_call,
        max_unsolved_depths=max_unsolved_depths,
    )
    res_depth.run_time += res_count.run_time
    return res_depth
