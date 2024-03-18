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

"""Clifford SAT problem."""

import numpy as np

from qiskit.quantum_info import Clifford, StabilizerState
from qiskit.synthesis.stabilizer.stabilizer_decompose import _calc_pauli_diff_stabilizer
from qiskit.synthesis.clifford.clifford_decompose_layers import _calc_pauli_diff
from qiskit.synthesis.permutation.permutation_utils import _inverse_pattern

from .sat_problem import SatProblem, SatProblemResult


class SatProblemClifford(SatProblem):
    def __init__(self, nq, verbosity=0):
        self.state_preparation_mode = False
        self.final_clifford = None
        super().__init__(nq, verbosity=verbosity)

    def set_init_matrix(self, matrix):
        self.init_state = matrix

    def set_init_matrix_to_identity(self, nq):
        self.init_state = np.eye(2 * nq)

    def set_final_clifford(self, clifford):
        self.final_clifford = clifford

    def set_state_preparation_mode(self, state_preparation_mode):
        self.state_preparation_mode = state_preparation_mode

    # Private methods, should not be used by the calling application

    def _encode_init_constraint(self, init_matrix_vars):
        """Encode constraints on the initial matrix."""
        n = 2 * self.nq

        if self.init_state is not None:
            for i in range(n):
                for j in range(n):
                    lit = (
                        init_matrix_vars[i, j]
                        if self.init_state[i, j] == 1
                        else -init_matrix_vars[i, j]
                    )
                    self.encoder.add_clause([lit])

    def _encode_final_constraint(self, matrix_vars):
        """Encode the constraint that the final matrix should satisfy.
        If state_preparation_mode is False, the final matrix should be equal to self.final_matrix.
        If state_preparation_mode is True, the final matrix should prepare the same state as self.final_matrix.
        """
        if self.final_clifford is not None:
            if not self.state_preparation_mode:
                self._encode_final_matrix_constraint(matrix_vars)
            else:
                self._encode_final_state_constraint(matrix_vars)

    def _encode_final_matrix_constraint(self, matrix_vars):
        """Encode the constraint that matrix_vars are equal to self.final_matrix."""
        symplectic_matrix = self.final_clifford.symplectic_matrix
        for i in range(2 * self.nq):
            for j in range(2 * self.nq):
                lit = (
                    matrix_vars[i, j]
                    if symplectic_matrix[i, j] == 1
                    else -matrix_vars[i, j]
                )
                self.encoder.add_clause([lit])

    def _encode_final_state_constraint(self, matrix_vars):
        """
        Encode the constraint that matrix_vars prepare the same state as self.final_matrix.
        Here we use the fact that two symplectic matrices M1 and M2 prepare the same state iff the stabilizer rows
        of M1 commute with the stabilizer rows of M2.
        """
        symplectic_matrix = self.final_clifford.symplectic_matrix
        nq = self.nq
        for i in range(nq, 2 * nq):
            for j in range(nq, 2 * nq):
                xor_clause = []
                for r in range(0, nq):
                    if symplectic_matrix[j, r]:
                        xor_clause.append(matrix_vars[i, r + nq])
                    if symplectic_matrix[j, r + nq]:
                        xor_clause.append(matrix_vars[i, r])
                self.encoder.encode_general_xor(xor_clause)

    def _encode_permutation(self, start_mat_vars, end_mat_vars, perm_mat_vars):
        """
        Encode the constraint that the end matrix is obtained by applying a permutation
        to the start matrix. Here, start_mat_vars and end_mat_vars are 2n x 2n, and
        perm_mat_vars is nxn.
        """
        nq = self.nq
        for i in range(nq):
            for j in range(nq):
                for s in range(2 * self.nq):
                    self.encoder.encode_EQ(
                        end_mat_vars[s, j],
                        start_mat_vars[s, i],
                        acts=[-perm_mat_vars[i, j]],
                    )
                    self.encoder.encode_EQ(
                        end_mat_vars[s, j + nq],
                        start_mat_vars[s, i + nq],
                        acts=[-perm_mat_vars[i, j]],
                    )

    def _create_state_vars(self):
        """Creates variables representing the current state
        (for Cliffords these are entries of the 2nq x 2nq symplectic matrix).
        """
        return self.encoder.create_mat_with_new_vars(2 * self.nq)

    def _encode_1q_unused(self, layer, start_mat_vars, end_mat_vars, i, used_var):
        """Qubit i is unused."""
        nq = self.nq
        for s in range(2 * nq):
            x_old = start_mat_vars[s, i]
            z_old = start_mat_vars[s, i + nq]
            x_new = end_mat_vars[s, i]
            z_new = end_mat_vars[s, i + nq]
            self.encoder.encode_EQ(x_new, x_old, acts=[used_var])
            self.encoder.encode_EQ(z_new, z_old, acts=[used_var])

    def _encode_1q_gate(self, layer, start_mat_vars, end_mat_vars, i, g):
        gate_var = layer.q1_vars[i, g]
        if g == "I":
            self.encode_1q_I(start_mat_vars, end_mat_vars, i, gate_var)
        elif g == "H":
            self.encode_1q_H(start_mat_vars, end_mat_vars, i, gate_var)
        elif g == "S":
            self.encode_1q_S(start_mat_vars, end_mat_vars, i, gate_var)
        elif g == "SH":
            self.encode_1q_SH(start_mat_vars, end_mat_vars, i, gate_var)
        elif g == "HS":
            self.encode_1q_HS(start_mat_vars, end_mat_vars, i, gate_var)
        elif g == "SHS":
            self.encode_1q_SHS(start_mat_vars, end_mat_vars, i, gate_var)

        else:
            assert False

    def encode_1q_I(self, start_mat_vars, end_mat_vars, i, gate_var):
        """Applying the identity gate on qubit i
        (same as not using qubit i at all).
        """
        # If the identity gate is applied, columns remain the same
        nq = self.nq
        for s in range(2 * nq):
            x_old = start_mat_vars[s, i]
            z_old = start_mat_vars[s, i + nq]
            x_new = end_mat_vars[s, i]
            z_new = end_mat_vars[s, i + nq]
            self.encoder.encode_EQ(x_new, x_old, acts=[-gate_var])
            self.encoder.encode_EQ(z_new, z_old, acts=[-gate_var])

    def encode_1q_H(self, start_mat_vars, end_mat_vars, i, gate_var):
        # If the Hadamard gate is applied, columns are swapped
        nq = self.nq
        for s in range(2 * nq):
            x_old = start_mat_vars[s, i]
            z_old = start_mat_vars[s, i + nq]
            x_new = end_mat_vars[s, i]
            z_new = end_mat_vars[s, i + nq]
            self.encoder.encode_EQ(x_new, z_old, acts=[-gate_var])
            self.encoder.encode_EQ(z_new, x_old, acts=[-gate_var])

    def encode_1q_S(self, start_mat_vars, end_mat_vars, i, gate_var):
        # applying phase gate: x_new = x_old, z_new = x_old XOR z_old
        nq = self.nq
        for s in range(2 * nq):
            x_old = start_mat_vars[s, i]
            z_old = start_mat_vars[s, i + nq]
            x_new = end_mat_vars[s, i]
            z_new = end_mat_vars[s, i + nq]
            self.encoder.encode_EQ(x_new, x_old, acts=[-gate_var])
            self.encoder.encode_XOR(z_new, x_old, z_old, acts=[-gate_var])

    def encode_1q_SH(self, start_mat_vars, end_mat_vars, i, gate_var):
        #   new col i = old col i XOR old col i+NQ
        #   new col i+NQ = old col i
        nq = self.nq
        for s in range(2 * nq):
            x_old = start_mat_vars[s, i]
            z_old = start_mat_vars[s, i + nq]
            x_new = end_mat_vars[s, i]
            z_new = end_mat_vars[s, i + nq]
            self.encoder.encode_XOR(x_new, x_old, z_old, acts=[-gate_var])
            self.encoder.encode_EQ(z_new, x_old, acts=[-gate_var])

    def encode_1q_HS(self, start_mat_vars, end_mat_vars, i, gate_var):
        # means: first h, then s
        #   new col i = old_col i + NQ
        #   new col i + NQ = old_col i XOR old_col i+NQ
        nq = self.nq
        for s in range(2 * nq):
            x_old = start_mat_vars[s, i]
            z_old = start_mat_vars[s, i + nq]
            x_new = end_mat_vars[s, i]
            z_new = end_mat_vars[s, i + nq]
            self.encoder.encode_EQ(x_new, z_old, acts=[-gate_var])
            self.encoder.encode_XOR(z_new, x_old, z_old, acts=[-gate_var])

    def encode_1q_SHS(self, start_mat_vars, end_mat_vars, i, gate_var):
        nq = self.nq
        for s in range(2 * nq):
            x_old = start_mat_vars[s, i]
            z_old = start_mat_vars[s, i + nq]
            x_new = end_mat_vars[s, i]
            z_new = end_mat_vars[s, i + nq]
            self.encoder.encode_XOR(x_new, x_old, z_old, acts=[-gate_var])
            self.encoder.encode_EQ(z_new, z_old, acts=[-gate_var])

    def _encode_2q_gate(self, layer, start_mat_vars, end_mat_vars, i, j, g):
        gate_var = layer.q2_vars[i, j, g]
        if g == "CX":
            self.encode_2q_CX(start_mat_vars, end_mat_vars, i, j, gate_var)
        elif g == "CZ":
            self.encode_2q_CZ(start_mat_vars, end_mat_vars, i, j, gate_var)
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
        # columnwise:
        #   x_j' = x_j XOR x_i
        #   z_i' = z_i XOR z_j
        #   end_mat[k] = start_mat[k] for k != j, i+NQ
        nq = self.nq
        for s in range(2 * nq):
            self.encoder.encode_XOR(
                end_mat_vars[s, j],
                start_mat_vars[s, j],
                start_mat_vars[s, i],
                acts=[-gate_var],
            )
            self.encoder.encode_XOR(
                end_mat_vars[s, i + nq],
                start_mat_vars[s, j + nq],
                start_mat_vars[s, i + nq],
                acts=[-gate_var],
            )
            self.encoder.encode_EQ(
                end_mat_vars[s, j + nq], start_mat_vars[s, j + nq], acts=[-gate_var]
            )
            self.encoder.encode_EQ(
                end_mat_vars[s, i], start_mat_vars[s, i], acts=[-gate_var]
            )

    def encode_2q_CZ(self, start_mat_vars, end_mat_vars, i, j, gate_var):
        """
        Encode act_lit -> (end_mat is obtained from start_mat by applying CZ(i, j))
        """
        # z_i' = z_i ^ x_j
        # z_j' = z_j ^ x_i
        # x_i' = x_i
        # x_j' = x_j
        nq = self.nq
        for s in range(2 * nq):
            self.encoder.encode_XOR(
                end_mat_vars[s, i + nq],
                start_mat_vars[s, i + nq],
                start_mat_vars[s, j],
                acts=[-gate_var],
            )
            self.encoder.encode_XOR(
                end_mat_vars[s, j + nq],
                start_mat_vars[s, j + nq],
                start_mat_vars[s, i],
                acts=[-gate_var],
            )
            self.encoder.encode_EQ(
                end_mat_vars[s, i], start_mat_vars[s, i], acts=[-gate_var]
            )
            self.encoder.encode_EQ(
                end_mat_vars[s, j], start_mat_vars[s, j], acts=[-gate_var]
            )

    def encode_2q_DCX(self, start_mat_vars, end_mat_vars, i, j, gate_var):
        """
        Encode act_lit -> (end_mat is obtained from start_mat by applying DCX(i, j))
        """
        # x_i' = x_j
        # x_j' = x_j ^ x_i
        # z_i' = z_j ^ z_i
        # z_j' = z_i

        nq = self.nq
        for s in range(2 * nq):
            self.encoder.encode_EQ(
                end_mat_vars[s, i], start_mat_vars[s, j], acts=[-gate_var]
            )
            self.encoder.encode_XOR(
                end_mat_vars[s, j],
                start_mat_vars[s, j],
                start_mat_vars[s, i],
                acts=[-gate_var],
            )
            self.encoder.encode_XOR(
                end_mat_vars[s, i + nq],
                start_mat_vars[s, j + nq],
                start_mat_vars[s, i + nq],
                acts=[-gate_var],
            )
            self.encoder.encode_EQ(
                end_mat_vars[s, j + nq], start_mat_vars[s, i + nq], acts=[-gate_var]
            )

    def encode_2q_SWAP(self, start_mat_vars, end_mat_vars, i, j, gate_var):
        """
        Encode act_lit -> (end_mat is obtained from start_mat by applying SWAP(i, j))
        """
        # z_i' = z_j
        # z_j' = z_i
        # x_i' = x_j
        # x_j' = x_i
        nq = self.nq
        for s in range(2 * nq):
            self.encoder.encode_EQ(
                end_mat_vars[s, i + nq], start_mat_vars[s, j + nq], acts=[-gate_var]
            )
            self.encoder.encode_EQ(
                end_mat_vars[s, j + nq], start_mat_vars[s, i + nq], acts=[-gate_var]
            )
            self.encoder.encode_EQ(
                end_mat_vars[s, i], start_mat_vars[s, j], acts=[-gate_var]
            )
            self.encoder.encode_EQ(
                end_mat_vars[s, j], start_mat_vars[s, i], acts=[-gate_var]
            )

    def _fix_returned_result(
        self, synthesis_result: SatProblemResult
    ) -> SatProblemResult:
        """
        For synthesizing a given Clifford:
            Given a Clifford circuit qc and the "target" clifford, we want to append
            a layer of Pauli gates to qc to match target's clifford phase.
        For state preparation:
            Given a Clifford circuit qc and the "target" clifford, we want to append
            a layer of Pauli gates to make sure all expected values are +1.
        """

        num_qubits = synthesis_result.circuit.num_qubits
        if synthesis_result.layout_permutation is not None:
            layout_permutation = synthesis_result.layout_permutation
        else:
            layout_permutation = list(range(num_qubits))

        if synthesis_result.final_permutation is not None:
            final_permutation = synthesis_result.final_permutation
        else:
            final_permutation = list(range(num_qubits))
        inverse_final_permutation = _inverse_pattern(final_permutation)

        checked_qc = synthesis_result.circuit_with_permutations
        cliff = Clifford(checked_qc)

        # Create the Pauli layer, but it needs to be remapped:
        # e.g. final_perm maps 3->5, adding Pauli x(5) after final_perm is equivalent to adding x(3) before final_perm

        permuted_qubits = [
            layout_permutation[inverse_final_permutation[q]] for q in range(num_qubits)
        ]
        permuted_qubits = _inverse_pattern(permuted_qubits)

        if not self.state_preparation_mode:
            pauli_circ = _calc_pauli_diff(cliff, self.final_clifford)
        else:
            pauli_circ = _calc_pauli_diff_stabilizer(cliff, self.final_clifford)

        synthesis_result.circuit.compose(
            pauli_circ, qubits=permuted_qubits, inplace=True
        )

        synthesis_result.circuit_with_permutations = (
            self._build_circuit_with_permutations(
                synthesis_result.circuit,
                synthesis_result.layout_permutation,
                synthesis_result.final_permutation,
            )
        )

        return synthesis_result

    def _check_returned_result(self, synthesis_result: SatProblemResult):
        """
        Check whether the obtained solution implements target_clifford
        """
        ok = True
        if self.final_clifford is not None:
            checked_qc = synthesis_result.circuit_with_permutations

            cliff = Clifford(checked_qc)
            ok = True

            if not self.state_preparation_mode:
                ok &= cliff == self.final_clifford

            else:
                stab = StabilizerState(cliff)
                stab_target = StabilizerState(cliff)
                ok &= stab.equiv(stab_target)
                ok &= stab.probabilities_dict() == stab_target.probabilities_dict()

        assert ok
        return ok
