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

"""Linear SAT problem."""

import numpy as np

from qiskit.circuit.library import LinearFunction
from .sat_problem import SatProblem, SatProblemResult


class SatProblemLinear(SatProblem):
    def __init__(self, nq, nc=0, verbosity=0):
        """nq must be the total number of qubits,"""
        self.init_matrix = None
        self.final_matrix = None
        super().__init__(nq, nc=nc, verbosity=verbosity)

    def set_init_matrix(self, matrix):
        self.init_matrix = matrix

    def set_init_matrix_to_identity(self, nq):
        self.init_matrix = np.eye(nq)

    def set_final_matrix(self, matrix):
        self.final_matrix = matrix

    def _encode_init_constraint(self, matrix_vars):
        """Encode constraints on the initial matrix."""
        if self.init_matrix is not None:
            for i in range(self.nq):
                for j in range(self.nq):
                    if self.init_matrix[i, j] == 1:
                        self.encoder.add_clause([matrix_vars[i, j]])
                    elif self.init_matrix[i, j] == 0:
                        self.encoder.add_clause([-matrix_vars[i, j]])

    def _encode_final_constraint(self, matrix_vars):
        """Encode the constraint that matrix_vars are equal to self.final_matrix."""
        if self.final_matrix is not None:
            for i in range(self.nq):
                for j in range(self.nq):
                    if self.final_matrix[i, j] == 1:
                        self.encoder.add_clause([matrix_vars[i, j]])
                    elif self.final_matrix[i, j] == 0:
                        self.encoder.add_clause([-matrix_vars[i, j]])

    def _encode_permutation(self, start_mat_vars, end_mat_vars, perm_mat_vars):
        """
        Encode the constraint that the end matrix is obtained by applying a permutation
        to the start matrix. Here, start_mat_vars and end_mat_vars are n x n, and
        perm_mat_vars is nxn.
        """
        nq = self.nq
        for i in range(nq):
            for j in range(nq):
                for s in range(self.nq):
                    self.encoder.encode_EQ(
                        end_mat_vars[j, s],
                        start_mat_vars[i, s],
                        acts=[-perm_mat_vars[i, j]],
                    )

    def _create_state_vars(self):
        return self.encoder.create_mat_with_new_vars(self.nq)

    def _compute_problem_vars(self):
        """Compute the list of main CNF variables that define the solution."""
        for layer in self.layers:
            self.problem_vars.extend(layer.q2_vars.values())

    def encode_1q_gate(self, layer, start_mat_vars, end_mat_vars, i, g):
        gate_var = layer.q1_vars[(i, g)]

        if g == "MEASUREMENT":
            self.encode_measurement(end_mat_vars, i, gate_var)

        else:
            assert False

    def encode_measurement(self, end_mat_vars, qbit, gate_var):
        """Measurement on qbit i simply forces the ith row to all 0s"""
        for j in range(self.nq):
            self.encoder.add_clause([-end_mat_vars[qbit, j]], acts=[-gate_var])

    def encode_2q_gate(self, layer, start_mat_vars, end_mat_vars, i, j, g):
        gate_var = layer.q2_vars[i, j, g]
        if g == "CX":
            self.encode_2q_CX(start_mat_vars, end_mat_vars, i, j, gate_var)
        elif g == "DCX":
            self.encode_2q_DCX(start_mat_vars, end_mat_vars, i, j, gate_var)
        elif g == "SWAP":
            self.encode_2q_SWAP(start_mat_vars, end_mat_vars, i, j, gate_var)
        else:
            assert False

    def encode_2q_CX(self, start_mat_vars, end_mat_vars, i, j, gate_var):
        """
        Encode act_lit -> (end_mat is obtained from start_mat by applying CX(i, j))
        """
        # CNOT(i, j) ==>
        #       (ith row)' = (ith row)
        #       (jth row)' = (jth row) ^ (ith row)
        for s in range(self.nq):
            self.encoder.encode_EQ(
                end_mat_vars[i, s], start_mat_vars[i, s], acts=[-gate_var]
            )
            self.encoder.encode_XOR(
                end_mat_vars[j, s],
                start_mat_vars[j, s],
                start_mat_vars[i, s],
                acts=[-gate_var],
            )

    def encode_2q_SWAP(self, start_mat_vars, end_mat_vars, i, j, gate_var):
        """
        Encode act_lit -> (end_mat is obtained from start_mat by applying SWAP(i, j))
        """
        # SWAP(i, j) ==>
        #       (ith row)' = (jth row)
        #       (jth row)' = (ith row)
        for s in range(self.nq):
            self.encoder.encode_EQ(
                end_mat_vars[i, s], start_mat_vars[j, s], acts=[-gate_var]
            )
            self.encoder.encode_EQ(
                end_mat_vars[j, s], start_mat_vars[i, s], acts=[-gate_var]
            )

    def encode_2q_DCX(self, start_mat_vars, end_mat_vars, i, j, gate_var):
        """
        Encode act_lit -> (end_mat is obtained from start_mat by applying DCX(i, j))
        """
        # CNOT(i, j) ==>
        #       (ith row)' = (jth row)
        #       (jth row)' = (jth row) ^ (ith row)
        for s in range(self.nq):
            self.encoder.encode_EQ(
                end_mat_vars[i, s], start_mat_vars[j, s], acts=[-gate_var]
            )
            self.encoder.encode_XOR(
                end_mat_vars[j, s],
                start_mat_vars[j, s],
                start_mat_vars[i, s],
                acts=[-gate_var],
            )

    def encode_1q_unused(self, layer, start_mat_vars, end_mat_vars, i, used_var):
        """Qubit i is unused."""
        nq = self.nq
        for s in range(nq):
            self.encoder.encode_EQ(
                end_mat_vars[i, s], start_mat_vars[i, s], acts=[used_var]
            )

    def fix_returned_result(self, result: SatProblemResult) -> SatProblemResult:
        return result

    def check_returned_result(self, result: SatProblemResult):
        # TODO: fix this for supporting clean ancillas
        ok = True
        if self.final_matrix is not None:
            linear_function = LinearFunction(result.circuit_with_permutations)
            ok = np.all(linear_function.linear == self.final_matrix)
        return ok
