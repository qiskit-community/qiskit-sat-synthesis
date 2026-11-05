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

"""Permutation SAT problem."""

from __future__ import annotations

import numpy as np

from qiskit.circuit.library import LinearFunction
from .sat_problem import SatProblem, SatProblemLayer, SatProblemResult


class SatProblemPermutation(SatProblem):
    def __init__(self, nq: int, verbosity: int = 0) -> None:
        self.final_matrix: np.ndarray | None = None
        super().__init__(nq, verbosity=verbosity)

    def set_init_matrix(self, matrix: np.ndarray) -> None:
        self.init_state = matrix

    def set_init_matrix_to_identity(self, nq: int) -> None:
        self.init_state = np.eye(nq)

    def set_final_matrix(self, matrix: np.ndarray) -> None:
        self.final_matrix = matrix

    def _encode_init_constraint(self, init_matrix_vars: np.ndarray) -> None:
        """Encode constraints on the initial matrix."""
        if self.init_state is not None:
            for i in range(self.nq):
                for j in range(self.nq):
                    lit = (
                        init_matrix_vars[i, j]
                        if self.init_state[i, j] == 1
                        else -init_matrix_vars[i, j]
                    )
                    self.encoder.add_clause([lit])

    def _encode_final_constraint(self, matrix_vars: np.ndarray) -> None:
        """Encode the constraint that matrix_vars are equal to self.final_matrix."""
        if self.final_matrix is not None:
            for i in range(self.nq):
                for j in range(self.nq):
                    lit = (
                        matrix_vars[i, j]
                        if self.final_matrix[i, j] == 1
                        else -matrix_vars[i, j]
                    )
                    self.encoder.add_clause([lit])

    def _create_state_vars(self) -> np.ndarray:
        return self.encoder.create_mat_with_new_vars(self.nq)

    def _compute_problem_vars(self) -> None:
        """Compute the list of main CNF variables that define the solution."""
        for layer in self.layers:
            self.problem_vars.extend(layer.q2_vars.values())

    def _encode_1q_gate(
        self,
        layer: SatProblemLayer,
        start_state_vars: np.ndarray,
        end_state_vars: np.ndarray,
        i: int,
        g: str,
    ) -> None:
        assert False
        pass

    def _encode_permutation(
        self,
        start_mat_vars: np.ndarray,
        end_mat_vars: np.ndarray,
        perm_mat_vars: np.ndarray,
    ) -> None:
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

    def _encode_2q_gate(
        self,
        layer: SatProblemLayer,
        start_mat_vars: np.ndarray,
        end_mat_vars: np.ndarray,
        i: int,
        j: int,
        g: str,
    ) -> None:
        gate_var = layer.q2_vars[i, j, g]
        if g == "SWAP":
            self.encode_2q_SWAP(start_mat_vars, end_mat_vars, i, j, gate_var)
        else:
            raise Exception(f"Gate {g} is not a supported 2-qubit gate.")

    def encode_2q_SWAP(
        self,
        start_mat_vars: np.ndarray,
        end_mat_vars: np.ndarray,
        i: int,
        j: int,
        gate_var: int,
    ) -> None:
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

    def _encode_1q_unused(
        self,
        layer: SatProblemLayer,
        start_mat_vars: np.ndarray,
        end_mat_vars: np.ndarray,
        i: int,
        used_var: int,
    ) -> None:
        """Qubit i is unused."""
        nq = self.nq
        for s in range(nq):
            self.encoder.encode_EQ(
                end_mat_vars[i, s], start_mat_vars[i, s], acts=[used_var]
            )

    def _fix_returned_result(self, result: SatProblemResult) -> SatProblemResult:
        return result

    def _check_returned_result(self, result: SatProblemResult) -> bool:
        ok = True
        if self.final_matrix is not None:
            linear_function = LinearFunction(result.circuit_with_permutations)
            ok = np.all(linear_function.linear == self.final_matrix)
        return ok
