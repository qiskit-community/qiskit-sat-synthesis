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
This file defines the partially abstract SatProblem class,
used for creating and encoding general problems into CNF,
solving them using a SAT-solver, and constructing circuits
from the solutions.
"""

from abc import abstractmethod

from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.permutation.permutation_utils import _get_ordered_swap

from .sat_encoder import SatEncoder, UnaryCounter
from .sat_solver import SatSolver, SolverStatus
from .sat_problem_constraints import (
    SatLayersIntersectOrOrderedConstraint,
    SatLayerNonEmptyConstraint,
    SatNonOverlapping2qConstraint,
    SatCommutation2Q1Q2QConstraint,
    SatMaxUnique2QLayersConstraint,
    SatCommutation2Q2QConstraint,
    Sat1QOnlyAfter2QConstraint,
    SatCannotSimplify2Q2QConstraint,
    SatCannotPush2QEarlierConstraint,
    SatCommutation2Q1Q2QForCountOnlyConstraint,
    SatCannotSimplify2Q1Q2QConstraint,
)


# LIST OF KNOWN 1Q-GATES:
_GATES_1Q = ["I", "S", "H", "SH", "HS", "SHS", "SQRTX", "MEASUREMENT"]

# LIST OF KNOWN 2Q-GATES
_GATES_2Q = ["CX", "CZ", "DCX", "SWAP"]


class SatProblemResult:
    """
    This class represents the result of solving a given sat problem.
    """

    def __init__(self):
        self.is_sat = False
        self.is_unsat = False
        self.is_unsolved = False

        # in case of a solution:
        self.solver_solution = None
        self.circuit = None
        self.layout_permutation = None
        self.final_permutation = None
        self.circuit_with_permutations = None
        self.num_1q = None
        self.num_2q = None

        self.run_time = 0

    def __repr__(self):
        if self.is_sat:
            out = (
                "found solution in "
                + str(round(self.run_time, 2))
                + " s; "
                + "num_1q = "
                + str(self.num_1q)
                + ", num_2q = "
                + str(self.num_2q)
            )
        else:
            out = "no solutions in " + str(round(self.run_time, 2)) + " s "
        return out


class SatProblemLayer:
    """
    Defines and encodes a single layer of the sat problem.
    """

    def __init__(
        self,
        sat_problem,
        gates,
        coupling_maps,
        max_num_1q_gates=None,
        max_num_2q_gates=None,
        full_2q=False,
    ):
        """
        Args:
          sat_problem: the sat problem containing this layer
          gates: gates allowed in this layer (any mixture of 1-qubit and 2-qubit gates)
          coupling_maps: list of coupling maps for this layer
          max_num_1q_gates: restriction on the maximum number of 1-qubit gates in this layer
          max_num_2q_gates: restriction on the maximum number of 2-qubit gates in this layer
          full_2q: option whether all the 2-qubit connections must be used
        """
        # sat problem containing this layer
        self.sat_problem = sat_problem

        # number of qubits
        self.nq = sat_problem.nq

        # number of classical bits
        self.nc = sat_problem.nc

        # List of 1-qubit gates allowed in this layer
        self.gates_1q = []

        # List of 2-qubit gates allowed in this layer
        self.gates_2q = []

        for gate in gates:
            gate_nq = self.sat_problem._num_qubits_in_gate(gate)
            if gate_nq == 1:
                self.gates_1q.append(gate)
            else:
                self.gates_2q.append(gate)

        # List of coupling maps allowed in this layer.
        # For most applications this consists of a single coupling map, in which case
        # we add the constraint that 2-qubit gates in the layer adhere to this coupling map.
        # For problems involving unique layers it is desirable to pass multiple coupling maps,
        # in which case we add the constraint that the connectivity of the layer adheres to
        # (at least) one of these coupling maps.
        self.coupling_maps = coupling_maps

        # max number of 1q gates in this layer
        self.max_num_1q_gates = max_num_1q_gates

        # max number of 2q gates in this layer
        self.max_num_2q_gates = max_num_2q_gates

        # In the case of a single coupling map, this option specifies whether all 2-qubit
        # connections must be used (ignoring the direction). In the case of multiple coupling
        # maps, this options specifies whether all 2-qubit connections of some coupling map
        # in the list must be used (ignoring the direction).
        self.full_2q = full_2q

        # q1_vars[i, g] is a CNF variable, which is true iff the layer contains
        # the 1-qubit gate g on qubit i
        self.q1_vars = {}

        # q2_vars[i, j, g] is a CNF variable, which is true iff the layer contains
        # the 2-qubit gate g with control i and target j
        self.q2_vars = {}

        # q_used_vars[i] is a CNF variable, which is true iff qubit i is used
        # in some 1-qubit or 2-qubit gate.
        self.q_used_vars = {}

        # connection_vars[i, j] is a CNF variable, which is true iff qubits i and j are
        # connected by a gate (i.e. there is a gate from qubit i to qubit j, or a gate from
        # qubit j to qubit i). Only pairs (i, j) for i < j are used.
        # This is optional, the value of None means that it has not been computed.
        self.connection_vars = None

        # in case there are multiple coupling maps, cmap_used_vars determine the
        # coupling map actually used in this layer
        self.cmap_used_vars = {}

    def is_connected(self, i, j):
        """Returns True if [i, j] is in one of the coupling maps."""
        for cmap in self.coupling_maps:
            if [i, j] in cmap:
                return True
            if (i, j) in cmap:
                return True
        return False

    def is_connected_k(self, i, j, k):
        """Returns True if [i, j] is in kth coupling map."""
        if [i, j] in self.coupling_maps[k]:
            return True
        if (i, j) in self.coupling_maps[k]:
            return True
        return False

    def _get_connection_var(self, i, j):
        """Returns connection_vars[i, j] when created."""
        if not self.connection_vars:
            return None
        return self.connection_vars.get((i, j), None)

    def _encode_connection_vars(self):
        """Encodes connection_vars logic."""
        if self.connection_vars is not None:
            return
        self.connection_vars = {}
        for i in range(self.nq):
            for j in range(i + 1, self.nq):
                if self.is_connected(i, j) or self.is_connected(j, i):
                    var_list = []
                    for g in self.gates_2q:
                        var = self.q2_vars.get((i, j, g), None)
                        if var is not None:
                            var_list.append(var)
                        var = self.q2_vars.get((j, i, g), None)
                        if var is not None:
                            var_list.append(var)
                    if len(var_list) == 1:
                        # If there is only one entry in the list, reuse it
                        self.connection_vars[i, j] = var_list[0]
                    elif len(var_list) >= 1:
                        new_var = self.sat_problem.encoder.new_var()
                        self.sat_problem.encoder.encode_EQ_OR(var_list, new_var)
                        self.connection_vars[i, j] = new_var

    def _process_solution(self, solution):
        """
        Retrieves the quantum circuit for this layer from sat-solver's solution.
        """
        qc = QuantumCircuit(self.nq, self.nc)
        num_1q = 0
        num_2q = 0

        # 1-qubit gates
        for i in range(self.nq):
            for g in self.gates_1q:
                var = self.q1_vars[i, g]

                if solution[var] == 1:
                    num_1q += 1
                    if g == "I":
                        pass
                    elif g == "S":
                        qc.s(i)
                    elif g == "H":
                        qc.h(i)
                    elif g == "SH":
                        qc.s(i)
                        qc.h(i)
                    elif g == "HS":
                        qc.h(i)
                        qc.s(i)
                    elif g == "SHS":
                        qc.s(i)
                        qc.h(i)
                        qc.s(i)
                    elif g == "SQRTX":
                        qc.h(i)
                        qc.s(i)
                        qc.h(i)
                    elif g == "MEASUREMENT":
                        qc.h(i)
                        qc.measure(i, i)
                    else:
                        assert (
                            False
                        ), f"Cannot reconstruct quantum circuit from gate {g}."

        # 2-qubit gates
        for i in range(self.nq):
            for j in range(self.nq):
                if self.is_connected(i, j):
                    for g in self.gates_2q:
                        var = self.q2_vars[i, j, g]

                        if solution[var] == 1:
                            num_2q += 1
                            if g == "CX":
                                qc.cx(i, j)
                            elif g == "CZ":
                                qc.cz(i, j)
                            elif g == "DCX":
                                qc.dcx(i, j)
                            elif g == "SWAP":
                                qc.swap(i, j)
                            else:
                                assert (
                                    False
                                ), f"Cannot reconstruct quantum circuit from gate {g}."
        return qc, num_1q, num_2q

    def _encode_coupling_map_constraints(self):
        """
        Adding constraints to support multiple coupling maps or fully connected layers.
        """

        encoder = self.sat_problem.encoder

        num_cmaps = len(self.coupling_maps)

        # Introduce cmaps_used_vars when there is more than 1 coupling map
        if num_cmaps > 1:
            for i in range(num_cmaps):
                self.cmap_used_vars[i] = encoder.new_var()

        if num_cmaps > 1:
            for i in range(self.nq):
                for j in range(self.nq):
                    if self.is_connected(i, j):
                        # For each connection [i, j] and gate g we add the constraint
                        #   Y_{i, j, g} -> [cmap_1 \/ ... \/ cmap_k]
                        # where the cmaps in the list are those that include [i, j].
                        # If in the future we will have variables connection_used[i, j],
                        # then we can also replace the above constraints by
                        #   connection_used[i, j] -> [cmap_1 \/ ... \/ cmap_k]
                        for g in self.gates_2q:
                            clause = [-self.q2_vars[i, j, g]]
                            for k in range(num_cmaps):
                                if self.is_connected_k(i, j, k):
                                    clause.append(self.cmap_used_vars[k])
                            encoder.add_clause(clause)

            encoder.encode_at_most_one(self.cmap_used_vars)

        if self.full_2q:
            for i in range(self.nq):
                for j in range(i + 1, self.nq):
                    for k in range(num_cmaps):
                        if self.is_connected_k(i, j, k) or self.is_connected_k(j, i, k):
                            clause = []
                            if num_cmaps > 1:
                                clause.append(-self.cmap_used_vars[k])
                            for g in self.gates_2q:
                                if (
                                    var := self.q2_vars.get(tuple([i, j, g]), None)
                                ) is not None:
                                    clause.append(var)
                                if (
                                    var := self.q2_vars.get(tuple([j, i, g]), None)
                                ) is not None:
                                    clause.append(var)
                            # print(f"{i = }, {j = }, {k = }, {clause = }, {self.cmap_used_vars = }, {self.q2_vars = }")
                            encoder.add_clause(clause)

    def _encode_layer_constraints(self):

        encoder = self.sat_problem.encoder

        # Create q2_vars CNF variables
        for i in range(self.nq):
            for j in range(self.nq):
                if self.is_connected(i, j):
                    for g in self.gates_2q:
                        self.q2_vars[i, j, g] = encoder.new_var()

        # Create q1_vars CNF variables
        for i in range(self.nq):
            for g in self.gates_1q:
                self.q1_vars[i, g] = encoder.new_var()

        # Create q_used_vars CNF variables
        for i in range(self.nq):
            self.q_used_vars[i] = encoder.new_var()

        # Constraint for all i: qubit i can be a part of at most one gate
        # Also, qubit i is used iff it's used in at least one gate
        for i in range(self.nq):
            vars_i = []
            # 2q-gates with i being control
            for j in range(self.nq):
                if self.is_connected(i, j):
                    for g in self.gates_2q:
                        vars_i.append(self.q2_vars[i, j, g])
            # 2q-gates with i being target
            for j in range(self.nq):
                if self.is_connected(j, i):
                    for g in self.gates_2q:
                        vars_i.append(self.q2_vars[j, i, g])
            # 1q-gates on i
            for g in self.gates_1q:
                vars_i.append(self.q1_vars[i, g])

            encoder.encode_at_most_one(vars_i)
            encoder.encode_EQ_OR(vars_i, self.q_used_vars[i])

        # Optional restriction on the maximal number of 1q-gates
        if self.max_num_1q_gates is not None:
            vars = []
            for i in range(self.nq):
                for g in self.gates_1q:
                    vars.append(self.q1_vars[i, g])
            encoder.encode_at_most_k(vars, self.max_num_1q_gates)

        # Optional restriction on the maximal number of 2q-gates
        if self.max_num_2q_gates is not None:
            vars = []
            for i in range(self.nq):
                for j in range(self.nq):
                    if self.is_connected(i, j):
                        for g in self.gates_2q:
                            vars.append(self.q2_vars[i, j, g])
            encoder.encode_at_most_k(vars, self.max_num_2q_gates)

        # Consistency constraints with multiple coupling maps
        self._encode_coupling_map_constraints()


class SatProblem:
    """
    Base class for representing problems of the form "does there exist a circuit
    consisting of N layers such that ...", converting them to conjunctive normal form,
    solving them with a SAT-solver, and extracting results.

    This class implements common functionality. Child classes are required to
    implement the abstract methods, and are also able to reimplement other methods.

    The public API consists of setting various options, creating the problem
    by appending layers and constraints, and solving.
    """

    def __init__(self, nq, nc=0, verbosity=0):
        self.nq = nq  # number of qubits
        self.nc = nc  # number of classical bits

        self.init_state = None
        self.layers = []

        self.encoder = SatEncoder(verbosity=verbosity)

        # to do : combine these into a single array
        self.init_state_vars = None
        self.state_vars = []

        # list of problem variables that define a solution
        self.problem_vars = []

        # layout and final permutations
        self.allow_layout_permutation = False
        self.allow_final_permutation = False
        self.layout_permutation_vars = None
        self.final_permutation_vars = None

        # additional constraints
        self.additional_constraints = []

        # Z3-solver
        self.solver = None

        # keep track of whether the formula has been encoded
        self.encoded = False

        # verbosity
        self.verbosity = verbosity

        # supporting constraints on total number of 2q and 1q gates
        self.max_num_1q_gates = None
        self.max_num_2q_gates = None
        self.gates_1q_counter = None
        self.gates_2q_counter = None

        self.last_result = None

        self.optimize_2q_gates = False
        self.optimize_1q_gates = False

        self.check_solutions = False
        self.print_solutions = False

        self.max_conflicts_per_call = None
        self.timeout_per_call = None

    # option setters

    def set_check_solutions(self, check_solutions):
        self.check_solutions = check_solutions

    def set_print_solutions(self, print_solutions):
        self.print_solutions = print_solutions

    def set_allow_layout_permutation(self, allow_layout_permutation):
        self.allow_layout_permutation = allow_layout_permutation

    def set_allow_final_permutation(self, allow_final_permutation):
        self.allow_final_permutation = allow_final_permutation

    def set_max_conflicts_per_call(self, max_conflicts_per_call):
        self.max_conflicts_per_call = max_conflicts_per_call

    def set_timeout_per_call(self, timeout_per_call):
        self.timeout_per_call = timeout_per_call

    def set_max_num_1q_gates(self, max_num_1q_gates):
        self.max_num_1q_gates = max_num_1q_gates

    def set_max_num_2q_gates(self, max_num_2q_gates):
        self.max_num_2q_gates = max_num_2q_gates

    def set_optimize_2q_gate(self, optimize_2q_gates):
        self.optimize_2q_gates = optimize_2q_gates

    def set_optimize_1q_gate(self, optimize_1q_gates):
        self.optimize_1q_gates = optimize_1q_gates

    # creating problem

    def add_layer(
        self,
        gates,
        coupling_maps,
        full_2q=False,
        max_num_1q_gates=None,
        max_num_2q_gates=None,
    ) -> int:
        layer = SatProblemLayer(
            self,
            gates=gates,
            coupling_maps=coupling_maps,
            full_2q=full_2q,
            max_num_1q_gates=max_num_1q_gates,
            max_num_2q_gates=max_num_2q_gates,
        )
        self.layers.append(layer)
        return len(self.layers) - 1

    def fix_1q_gate_in_layer(self, layer_id, i, g):
        """Add the constraint that we have the 1-q gate g on qubit i in the given layer."""
        layer = self.layers[layer_id]
        assert (
            g in layer.gates_1q
        ), f"Gate {g} is not in the list of gates allowed in layer {layer_id}."
        self.encoder.add_clause([layer.q1_vars[i, g]])

    def exclude_1q_gate_in_layer(self, layer_id, i, g):
        """Add the constraint that we do not have the 1-q gate g on qubit i in the given layer."""
        layer = self.layers[layer_id]
        assert (
            g in layer.gates_1q
        ), f"Gate {g} is not in the list of gates allowed in layer {layer_id}."
        self.encoder.add_clause([-layer.q1_vars[i, g]])

    def fix_2q_gate_in_layer(self, layer_id, i, j, g):
        """Add the constraint that we have the 2-q gate g on qubits i,j in the given layer."""
        layer = self.layers[layer_id]
        assert (
            g in layer.gates_2q
        ), f"Gate {g} is not in the list of gates allowed in layer {layer_id}."
        assert layer.is_connected(
            i, j
        ), f"Qubits {i} and {j} are not connected in layer {layer_id}."
        self.encoder.add_clause([layer.q2_vars[i, j, g]])

    def exclude_2q_gate_in_layer(self, layer_id, i, j, g):
        """Add the constraint that we cannot have the 2-q gate g on qubits i,j in the given layer."""
        layer = self.layers[layer_id]
        assert (
            g in layer.gates_2q
        ), f"Gate {g} is not in the list of gates allowed in layer {layer_id}."
        assert layer.is_connected(
            i, j
        ), f"Qubits {i} and {j} are not connected in layer {layer_id}."
        self.encoder.add_clause([-layer.q2_vars[i, j, g]])

    # additional constraints

    def add_1q_only_after_2q_constraint(self, layer2q_id, layer1q_id):
        self.additional_constraints.append(
            Sat1QOnlyAfter2QConstraint(layer2q_id, layer1q_id)
        )

    def add_nonempty_constraint(self, layer_id):
        self.additional_constraints.append(SatLayerNonEmptyConstraint(layer_id))

    def add_max_unique_layers_constraint(self, layer_ids, max_unique):
        self.additional_constraints.append(
            SatMaxUnique2QLayersConstraint(layer_ids, max_unique)
        )

    def add_cannot_push_2q_earlier_constraint(self, prev_layer_id, next_layer_id):
        self.additional_constraints.append(
            SatCannotPush2QEarlierConstraint(prev_layer_id, next_layer_id)
        )

    def add_layers_intersect_or_ordered(self, layer1_id, layer2_id):
        self.additional_constraints.append(
            SatLayersIntersectOrOrderedConstraint(layer1_id, layer2_id)
        )

    def add_cannot_simplify_2q_2q_constraint(self, layer1_id, layer2_id):
        self.additional_constraints.append(
            SatCannotSimplify2Q2QConstraint(layer1_id, layer2_id)
        )

    def add_cannot_simplify_2q_1q_2q_constraint(self, layer1, layer2, layer3):
        self.additional_constraints.append(
            SatCannotSimplify2Q1Q2QConstraint(layer1, layer2, layer3)
        )

    def add_commutation_2q_2q_constraint(self, layer1_id, layer2_id):
        self.additional_constraints.append(
            SatCommutation2Q2QConstraint(layer1_id, layer2_id)
        )

    def add_commutation_2q_1q_2q_constraint(self, layer1_id, layer2_id, layer3_id):
        self.additional_constraints.append(
            SatCommutation2Q1Q2QConstraint(layer1_id, layer2_id, layer3_id)
        )

    def add_commutation_2q_1q_2q_count_only_constraint(
        self, layer1_id, layer2_id, layer3_id
    ):
        self.additional_constraints.append(
            SatCommutation2Q1Q2QForCountOnlyConstraint(layer1_id, layer2_id, layer3_id)
        )

    def add_nonoverlapping_2q_constraint(self, layer_id):
        self.additional_constraints.append(SatNonOverlapping2qConstraint(layer_id))

    # solving

    def solve(self) -> SatProblemResult:
        if self.verbosity >= 1:
            print(f" => synthesis problem has {len(self.layers)} layers")
        self._encode()
        self.solver = SatSolver(
            self.encoder, assumptions_as_unit_clauses=False, verbosity=self.verbosity
        )
        sat_query_result = self.solver.solve(
            max_conflicts=self.max_conflicts_per_call, timeout=self.timeout_per_call
        )
        result = self._extract_sat_problem_result(sat_query_result)
        self.last_result = result
        if self.verbosity >= 1:
            print(f" => synthesis problem result: {self.last_result}")
        if result.is_sat:
            if self.optimize_2q_gates:
                result = self._optimize_num_2q_gates(max_num_1q_gates=None)
            if self.optimize_1q_gates:
                num_2q = result.num_2q
                result = self._optimize_num_1q_gates(max_num_2q_gates=num_2q)

            result = self._fix_returned_result(result)
            if self.check_solutions:
                if self.verbosity >= 1:
                    print(f"Checking solution")

                ok = self._check_returned_result(result)
                if self.verbosity >= 1:
                    print(f" => solution is correct: {ok}")

            if self.print_solutions:
                print(f"Printing solution")
                print(result.circuit)
        return result

    def solve_another(self) -> SatProblemResult:
        assert self.last_result is not None
        assert self.last_result.solver_solution is not None
        self.encoder.block_solution(
            self.last_result.solver_solution, blocking_vars=self.problem_vars
        )
        sat_query_result = self.solver.solve(
            max_conflicts=self.max_conflicts_per_call, timeout=self.timeout_per_call
        )
        result = self._extract_sat_problem_result(sat_query_result)
        self.last_result = result
        if self.verbosity >= 1:
            print(f" => additional synthesis problem result: {self.last_result}")
        if result.is_sat:
            result = self._fix_returned_result(result)
            if self.check_solutions:
                if self.verbosity >= 1:
                    print(f"Checking solution")
                ok = self._check_returned_result(result)
                if self.verbosity >= 1:
                    print(f" => solution is correct: {ok}")
            if self.print_solutions:
                print(f"Printing solution")
                print(result.circuit)
        return result

    # internal methods

    def _extend_max_gates_counters(self, max_num_1q_gates, max_num_2q_gates):
        if max_num_1q_gates is not None:
            if self.gates_1q_counter is None:
                # create a list of all 1-qubit gates
                vars_1q = []
                for layer in self.layers:
                    vars_1q.extend(layer.q1_vars.values())
                self.gates_1q_counter = UnaryCounter(vars_1q, self.encoder)

            self.gates_1q_counter.extend(max_num_1q_gates + 1)

        if max_num_2q_gates is not None:
            if self.gates_2q_counter is None:
                # create a list of all 2-qubit gates
                vars_2q = []
                for layer in self.layers:
                    vars_2q.extend(layer.q2_vars.values())
                self.gates_2q_counter = UnaryCounter(vars_2q, self.encoder)

            self.gates_2q_counter.extend(max_num_2q_gates + 1)

    def _encode_max_gates_constraints(self):
        self._extend_max_gates_counters(self.max_num_1q_gates, self.max_num_2q_gates)

        if self.max_num_1q_gates is not None:
            counter_var = self.gates_1q_counter.get_counter_var(
                self.max_num_1q_gates + 1
            )
            if counter_var is not None:
                self.encoder.add_clause([-counter_var])

        if self.max_num_2q_gates is not None:
            counter_var = self.gates_2q_counter.get_counter_var(
                self.max_num_2q_gates + 1
            )
            if counter_var is not None:
                self.encoder.add_clause([-counter_var])

    def _restrict_max_gates(self, max_num_1q_gates, max_num_2q_gates):
        self._extend_max_gates_counters(max_num_1q_gates, max_num_2q_gates)

        if max_num_1q_gates is not None:
            counter_var = self.gates_1q_counter.get_counter_var(max_num_1q_gates + 1)
            if counter_var is not None:
                self.encoder.add_assumption(-counter_var)

        if max_num_2q_gates is not None:
            counter_var = self.gates_2q_counter.get_counter_var(max_num_2q_gates + 1)
            if counter_var is not None:
                self.encoder.add_assumption(-counter_var)

    def _compute_problem_vars(self):
        """Compute the list of main CNF variables that define the solution."""
        for layer in self.layers:
            self.problem_vars.extend(layer.q1_vars.values())
            self.problem_vars.extend(layer.q2_vars.values())
        # self.problem_vars.extend(self.layout_permutation_vars)
        # self.problem_vars.extend(self.final_permutation_vars)

    def _extract_sat_problem_result(self, sat_query_result) -> SatProblemResult:
        """Extract solution (quantum circuit and layouts) from SAT-solver assignment."""
        solver_result = sat_query_result.result
        solver_solution = sat_query_result.solution
        solver_run_time = sat_query_result.run_time

        res = SatProblemResult()
        res.run_time = solver_run_time
        res.solver_solution = solver_solution

        if solver_result == SolverStatus.SAT:
            assert solver_solution is not None
            nq = self.nq
            nc = self.nc
            qc = QuantumCircuit(nq, nc)
            layout_permutation = None
            final_permutation = None

            if self.allow_layout_permutation:
                layout_permutation = self.encoder.get_perm_pattern_from_solution(
                    self.layout_permutation_vars, solver_solution
                )

            if self.allow_final_permutation:
                final_permutation = self.encoder.get_perm_pattern_from_solution(
                    self.final_permutation_vars, solver_solution
                )

            res.num_1q = 0
            res.num_2q = 0

            init_state_circuit = self._process_solution_init_state(solver_solution)
            if init_state_circuit is not None:
                qc.append(init_state_circuit, range(self.nq), range(nc))
                qc.barrier()

            for layer in self.layers:
                layer_circuit, layer_1q, layer_2q = layer._process_solution(
                    solver_solution
                )
                qc.append(layer_circuit, range(nq), range(nc))
                qc.barrier()
                res.num_1q += layer_1q
                res.num_2q += layer_2q

            qc = qc.decompose()

            res.is_sat = True
            res.circuit = qc
            res.layout_permutation = layout_permutation
            res.final_permutation = final_permutation
            res.circuit_with_permutations = self._build_circuit_with_permutations(
                qc, layout_permutation, final_permutation
            )

        elif solver_result == SolverStatus.UNSAT:
            res.is_unsat = True

        else:
            res.is_unsolved = True

        return res

    def _build_circuit_with_permutations(
        self, qc, layout_permutation, final_permutation
    ):
        """Creates a FULL circuit from synthesized circuit by explicitly appending permutations onto it."""

        if layout_permutation is None and final_permutation is None:
            return qc.copy()
        checked_qc = QuantumCircuit(qc.num_qubits, qc.num_clbits)

        # add initial permutation
        if layout_permutation:
            swap_list = _get_ordered_swap(layout_permutation)
            for swap_pair in swap_list:
                checked_qc.swap(swap_pair[0], swap_pair[1])

        for inst, qargs, cargs in qc.data:
            checked_qc.append(inst, qargs, cargs)

        # add inverse of initial permutation
        if layout_permutation:
            swap_list = _get_ordered_swap(layout_permutation)
            swap_list.reverse()

            for swap_pair in swap_list:
                checked_qc.swap(swap_pair[0], swap_pair[1])

        # add final permutation
        if final_permutation:
            swap_list = _get_ordered_swap(final_permutation)
            for swap_pair in swap_list:
                checked_qc.swap(swap_pair[0], swap_pair[1])

        return checked_qc

    def _optimize_num_2q_gates(self, max_num_1q_gates):
        if self.last_result is None:
            return self.last_result

        if not self.last_result.is_sat:
            return None

        if self.verbosity >= 1:
            print(
                f"Optimizing the number of 2q-gates (restricting the number of 1q-gates to {max_num_1q_gates})"
            )

        while True:
            if self.last_result.num_2q == 0:
                break
            if self.verbosity >= 1:
                print(
                    f" => looking for solutions with num_1q = {max_num_1q_gates}, num_2q = {self.last_result.num_2q-1}"
                )
            self._restrict_max_gates(max_num_1q_gates, self.last_result.num_2q - 1)
            new_query_result = self.solver.solve(
                max_conflicts=self.max_conflicts_per_call, timeout=self.timeout_per_call
            )
            self.encoder.clear_assumptions()

            if new_query_result.result == SolverStatus.UNSAT:
                break
            elif new_query_result.result != SolverStatus.SAT:
                break

            new_result = self._extract_sat_problem_result(new_query_result)
            assert new_result.num_2q <= self.last_result.num_2q - 1

            # update the result
            self.last_result = new_result

            if self.verbosity >= 1:
                print(f" => improved synthesis problem result: {self.last_result}")

        return self.last_result

    def _optimize_num_1q_gates(self, max_num_2q_gates):
        if self.last_result is None:
            return self.last_result

        if not self.last_result.is_sat:
            return None

        if self.verbosity >= 1:
            print(
                f"Optimizing the number of 1q-gates (restricting the number of 2q-gates to {max_num_2q_gates})"
            )

        while True:
            if self.last_result.num_1q == 0:
                break
            if self.verbosity >= 1:
                print(
                    f" => looking for solutions with num_1q = {self.last_result.num_1q-1}, num_2q = {max_num_2q_gates}"
                )
            self._restrict_max_gates(self.last_result.num_1q - 1, max_num_2q_gates)
            new_query_result = self.solver.solve(
                max_conflicts=self.max_conflicts_per_call, timeout=self.timeout_per_call
            )
            self.encoder.clear_assumptions()

            if new_query_result.result == SolverStatus.UNSAT:
                break
            elif new_query_result.result != SolverStatus.SAT:
                break

            new_result = self._extract_sat_problem_result(new_query_result)
            assert new_result.num_1q <= self.last_result.num_1q - 1

            # update the result
            self.last_result = new_result

            if self.verbosity >= 1:
                print(f" => improved synthesis problem result: {self.last_result}")

        return self.last_result

    def _encode_transition_relation(self, layer, start_state_vars, end_state_vars):
        # For each 2-qubit CNF variable [i, j, g], encode the update state
        # constraint for qubits i, j.
        #   i.e. Y_[i, j, g] => (how end_state_vars are related to start_state_vars)
        for i in range(self.nq):
            for j in range(self.nq):
                if layer.is_connected(i, j):
                    for g in layer.gates_2q:
                        self._encode_2q_gate(
                            layer, start_state_vars, end_state_vars, i, j, g
                        )

        # For each 1-qubit CNF variable [i, g], encode the update state constraint on
        # qubit i.
        for i in range(self.nq):
            for g in layer.gates_1q:
                self._encode_1q_gate(layer, start_state_vars, end_state_vars, i, g)

        # For each qubit i, if it is not used in any gate, then the
        # state is not changed.
        for i in range(self.nq):
            self._encode_1q_unused(
                layer, start_state_vars, end_state_vars, i, layer.q_used_vars[i]
            )

    def _encode(self):
        if self.encoded:
            return

        nq = self.nq

        # initial constraint
        cur_state_vars = self._create_state_vars()
        self._encode_init_constraint(cur_state_vars)
        self.init_state_vars = cur_state_vars

        if self.allow_layout_permutation:
            perm_mat_vars = self.encoder.create_perm_mat_with_new_vars(nq)
            end_state_vars = self._create_state_vars()
            self._encode_permutation(cur_state_vars, end_state_vars, perm_mat_vars)
            cur_state_vars = end_state_vars
            self.layout_permutation_vars = perm_mat_vars

        for layer in self.layers:
            end_state_vars = self._create_state_vars()
            layer._encode_layer_constraints()
            self._encode_transition_relation(layer, cur_state_vars, end_state_vars)
            cur_state_vars = end_state_vars
            self.state_vars.append(end_state_vars)

        if self.allow_layout_permutation:
            inv_perm_mat_vars = self.layout_permutation_vars.transpose()
            end_state_vars = self._create_state_vars()
            self._encode_permutation(cur_state_vars, end_state_vars, inv_perm_mat_vars)
            cur_state_vars = end_state_vars

        if self.allow_final_permutation:
            perm_mat_vars = self.encoder.create_perm_mat_with_new_vars(nq)
            end_state_vars = self._create_state_vars()
            self._encode_permutation(cur_state_vars, end_state_vars, perm_mat_vars)
            cur_state_vars = end_state_vars
            self.final_permutation_vars = perm_mat_vars

        self._encode_final_constraint(cur_state_vars)
        self._encode_max_gates_constraints()
        self._encode_additional_constraints()
        self._compute_problem_vars()
        self.encoded = True

    def _encode_additional_constraints(self):
        for constraint in self.additional_constraints:
            constraint._encode(self)

    def _process_solution_init_state(self, solution):
        return None

    def _num_qubits_in_gate(self, gate):
        """Returns number of qubits used for this gate (supported values are 1 in 2)"""
        if gate in _GATES_1Q:
            return 1
        elif gate in _GATES_2Q:
            return 2
        else:
            raise Exception(f"Gate {gate} is not in the lists of known gates.")

    # Methods that child classes have to implement

    @abstractmethod
    def _create_state_vars(self):
        raise NotImplementedError

    @abstractmethod
    def _encode_init_constraint(self, init_state_vars):
        raise NotImplementedError

    @abstractmethod
    def _encode_permutation(self, start_state_vars, end_state_vars, perm_mat_vars):
        raise NotImplementedError

    @abstractmethod
    def _encode_final_constraint(self, final_state_vars):
        raise NotImplementedError

    @abstractmethod
    def _encode_1q_gate(self, layer, start_state_vars, end_state_vars, i, g):
        raise NotImplementedError

    @abstractmethod
    def _encode_2q_gate(self, layer, start_state_vars, end_state_vars, i, j, g):
        raise NotImplementedError

    @abstractmethod
    def _encode_1q_unused(self, layer, start_state_vars, end_state_vars, i, used_var):
        raise NotImplementedError

    @abstractmethod
    def _fix_returned_result(self, result: SatProblemResult) -> SatProblemResult:
        raise NotImplementedError

    @abstractmethod
    def _check_returned_result(self, result: SatProblemResult):
        raise NotImplementedError
