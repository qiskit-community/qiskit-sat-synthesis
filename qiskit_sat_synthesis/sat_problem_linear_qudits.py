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

"""Linear SAT problem with qudits instead of qubits."""

import numpy as np
from qiskit_sat_synthesis.sat_problem import SatProblem, SatProblemResult
from qiskit_sat_synthesis.sat_solver import SolverStatus

# CNF generation notes (old version):
#
# At each point of time, we have the "current matrix" variables M[i, j, d] representing
# whether the entry in (i, j) is equal to d, where 0 <= i < nq, 0 <= j < nq, 0 <= d < qd.
#
# To represent transitions, we use the variables Y[i, j, c], representing whether
# row i is added to row j with multiple c, where 0 <= i < nq, 0 <= j < nq, 0 <= c < qd.
#
# Initial constraint: initially the matrix is identity, that is:
#   M[i, i, 1] for all i
#   M[i, j, 0] for all i != j
#
# Additional constraints on M:
# For each (i, j), there is exactly one number in (i, j): ExactlyOne { M[i, j, 0], ..., M[i, j, d-1] }.
#
# Final constraint: at the end the matrix coincides with the given target matrix:
#  If the entry in TGT[i, j] is d then add the unit clause M[i, j, d]..
#
# For transitions:
#
# First, we want to say that each qubit can be "used at most once", either when
# used as a control qubit or as a target qubit.
#
# For all i:
#   AtMost1 { Y[i, j, c], Y[j, i, c], with 0 <= i < nq, 0 <= c < qd}.
#
# This includes saying that we can use row i for adding at most once, we can add
# to row i at most once, and the same qubit can not be both control and target at
# the same time.
#
# We should not be able to add row i to itself, that is Y[i, i, c] = 0 for all i and c.
#   (ok, this is not needed, since this should be prohibited also by connectivity constraints).
#
# Connectivity constraints: if i is not connected to j, then Y[i, j, c] = 0 for all c.
#
# Introduce T[j] representing "row j got updated", for all j
#   T[j] == One of {Y[i, j, c], 0 <= i < nq, 0 <= c < qd }.
#
# The two main constraints are:
#   (1) If Y[i, j, c] = 1, then (row j)' becomes c * (row i) + row_j.
#   (2) if T[j] = 0, then (row j)' equals (row j).
#
# That is, (1) says what (row j) becomes if things get added to it, and (2) says what happens when
# nothing gets added to it.
#
# So, how to we encode "if Y[i, j, c] = 1, then (row j)' == c * (row i) + row_j"?
#
# Abusing notation, for each column s, we want M'[j, s] == c * M[i, s] + M[j, s].
#
# This should work (but requires a large number of clauses)
#
#   for i = 0, ..., nq
#     for j = 0, ..., nq
#       if i is not connected to j in the coupling map, break
#       for c = 0, ..., qd-1
#         for s = 0, ..., nq
#           for p = 0, ..., qd-1
#             for q = 0, ..., qd-1
#               compute r = c * p + q mod qd
#               if Y[i, j, c] and M[i, s, p] and M[j, s, q] then M'[j, s, r].
#
# The last line is a clause. Also, all possible values of r are covered (by varying q),
# so we do not need the "other" direction.


class SatProblemLinearQudits(SatProblem):
    def __init__(self, nq, qd, verbosity=0):
        self.final_matrix = None
        self.qd = qd
        super().__init__(nq, verbosity=verbosity)

    # def set_init_matrix(self, matrix):
    #     self.init_state = matrix
    #
    # def set_init_matrix_to_identity(self, nq):
    #     self.init_state = np.eye(nq)

    def set_final_matrix(self, matrix):
        self.final_matrix = matrix

    def _create_state_vars(self):
        """Create M[i, j, d] vars."""

        M = {}
        for i in range(self.nq):
            for j in range(self.nq):
                for d in range(self.qd):
                    M[i, j, d] = self.encoder.new_var()

        for i in range(self.nq):
            for j in range(self.nq):
                lits = [M[i, j, d] for d in range(self.qd)]
                self.encoder.encode_exactly_one(lits)

        return M

    def _encode_init_constraint(self, init_matrix_vars):
        """Encode constraints on the initial matrix."""
        for i in range(self.nq):
            for j in range(self.nq):
                if i == j:
                    self.encoder.add_clause([init_matrix_vars[i, j, 1]])
                else:
                    self.encoder.add_clause([init_matrix_vars[i, j, 0]])

    def _encode_final_constraint(self, matrix_vars):
        """Encodes final constraints on M, based on the self.tgt matrix."""
        for i in range(self.nq):
            for j in range(self.nq):
                for d in range(self.qd):
                    if self.final_matrix[i, j] == d:
                        self.encoder.add_clause([matrix_vars[i, j, d]])

    def _encode_permutation(self, start_mat_vars, end_mat_vars, perm_mat_vars):
        """
        Encode the constraint that the end matrix is obtained by applying a permutation
        to the start matrix.
        """
        raise Exception("Permutations are currently not allowed with qudits")

    def _compute_problem_vars(self):
        """Compute the list of main CNF variables that define the solution."""
        for layer in self.layers:
            self.problem_vars.extend(layer.q2_vars.values())

    def encode_1q_gate(self, layer, start_state_vars, end_state_vars, i, g):
        assert False
        pass

    def encode_2q_gate(self, layer, start_mat_vars, end_mat_vars, i, j, g):
        gate_var = layer.q2_vars[i, j, g]
        if g[0] == "CX":
            self.encode_2q_CX(start_mat_vars, end_mat_vars, i, j, g[1], gate_var)
        else:
            assert False

    def encode_2q_CX(self, start_mat_vars, end_mat_vars, i, j, c, gate_var):
        """
        Encode act_lit -> (end_mat is obtained from start_mat by applying CX(i, j))
        """
        # CNOT(i, j, c) ==>
        #       (ith row)' = (ith row)
        #       (jth row)' = (jth row) + c * (ith row) mod qd

        for s in range(self.nq):
            for p in range(self.qd):
                for q in range(self.qd):
                    r = (c * p + q) % self.qd
                    self.encoder.add_clause(
                        [
                            -start_mat_vars[i, s, p],
                            -start_mat_vars[j, s, q],
                            end_mat_vars[j, s, r],
                        ],
                        acts=[-gate_var],
                    )
                self.encoder.encode_EQ(
                    start_mat_vars[i, s, p], end_mat_vars[i, s, p], acts=[-gate_var]
                )

    def encode_1q_unused(self, layer, start_mat_vars, end_mat_vars, i, used_var):
        """Qubit i is unused."""
        nq = self.nq
        for s in range(nq):
            for p in range(self.qd):
                self.encoder.encode_EQ(
                    end_mat_vars[i, s, p], start_mat_vars[i, s, p], acts=[used_var]
                )

    def fix_returned_result(self, result: SatProblemResult) -> SatProblemResult:
        return result

    def check_returned_result(self, result: SatProblemResult):
        ok = True
        return ok

    def num_qubits_in_gate(self, gate):
        """Returns number of qubits used for this gate (supported values are 1 in 2)"""
        if isinstance(gate, tuple) and gate[0] == "CX":
            return 2
        else:
            raise Exception(f"Gate {gate} is not in the lists of known gates.")

    def _extract_sat_problem_result(self, sat_query_result) -> SatProblemResult:
        def print_mvars(matrix_vars):
            mat = np.zeros((self.nq, self.nq))
            for i in range(self.nq):
                for j in range(self.nq):
                    for c in range(self.qd):
                        if solver_solution[matrix_vars[i, j, c]]:
                            mat[i, j] = c
                            break
            print(mat.astype(int))

        def print_gate_vars(layer):
            # 2-qubit gates
            for i in range(self.nq):
                for j in range(self.nq):
                    if layer.is_connected(i, j):
                        for g in layer.gates_2q:
                            var = layer.q2_vars[i, j, g]
                            if solver_solution[var] == 1:
                                # num_2q += 1
                                c = g[1]
                                print(f"Row{j} += {c} * Row{i}")

        solver_result = sat_query_result.result
        solver_solution = sat_query_result.solution

        res = SatProblemResult()
        res.solver_solution = solver_solution
        res.run_time = sat_query_result.run_time

        if solver_result == SolverStatus.SAT:
            assert solver_solution is not None
            res.num_1q = 0
            res.num_2q = 0
            print("")
            print("PRINTING SOLUTION:")
            print("Initial Matrix")
            print_mvars(self.init_state_vars)
            print("===========")

            for k, layer in enumerate(self.layers):
                print("Transition:")
                print_gate_vars(layer)
                print("===========")
                print("Obtained matrix:")
                print_mvars(self.state_vars[k])
                print("===========")

            res.is_sat = True
            res.circuit = "CIRCUIT"
            res.layout_permutation = None
            res.final_permutation = None
            res.circuit_with_permutations = "CIRCUIT"
            res.k = len(self.layers)

        elif solver_result == SolverStatus.UNSAT:
            res.is_unsat = True

        else:
            res.is_unsolved = True

        return res
