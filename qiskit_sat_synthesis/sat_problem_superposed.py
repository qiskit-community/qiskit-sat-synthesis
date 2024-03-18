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

"""Synthesis of superposed states."""

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import StabilizerState
from .sat_problem import SatProblem, SatProblemResult


class SatProblemSuperposed(SatProblem):
    """
    Synthesis of superposed states of the form 1/root(2) |0> + 1/root(2) |s>.

    The circuit consists of a single Hadamard gate followed by a sequence of CX-gates.
    """

    def __init__(self, nq, verbosity=0):
        self.final_state = None
        self.root = None  # either a single qubit or a list of qubits
        super().__init__(nq, verbosity=verbosity)

    def set_root(self, root):
        self.root = root

    def set_final_matrix(self, final_state):
        self.final_state = final_state

    def _encode_init_constraint(self, init_state_vars):
        """Encodes initial constraint."""
        self.encoder.encode_at_most_one(init_state_vars)
        self.encoder.add_clause(init_state_vars)

        if self.root is not None:
            if isinstance(self.root, list):
                self.encoder.add_clause([init_state_vars[x] for x in self.root])
            else:
                self.encoder.add_clause([init_state_vars[self.root]])

    def _encode_final_constraint(self, state_vars):
        """Encodes final constraint, i.e. the final state is the one we target."""
        if self.final_state is not None:
            for i in range(self.nq):
                if self.final_state[i] == 1:
                    self.encoder.add_clause([state_vars[i]])
                else:
                    self.encoder.add_clause([-state_vars[i]])

    def set_allow_layout_permutation(self, allow_layout_permutation):
        if allow_layout_permutation is True:
            raise Exception(
                "Option allow_layout_permutation is currently not supported for superposed states."
            )

    def _encode_permutation(self, start_state_vars, end_state_vars, perm_mat_vars):
        """
        Encode the constraint that the end state is obtained by applying a permutation
        to the start state.
        """
        nq = self.nq
        for i in range(nq):
            for j in range(nq):
                self.encoder.encode_EQ(
                    end_state_vars[j], start_state_vars[i], acts=[-perm_mat_vars[i, j]]
                )

    def _create_state_vars(self):
        return [self.encoder.new_var() for _ in range(self.nq)]

    def _compute_problem_vars(self):
        """Compute the list of main CNF variables that define the solution."""
        for layer in self.layers:
            self.problem_vars.extend(layer.q2_vars.values())

    def _encode_1q_gate(self, layer, start_state_vars, end_state_vars, i, g):
        assert False
        pass

    def _encode_2q_gate(self, layer, start_mat_vars, end_mat_vars, i, j, g):
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
        self.encoder.encode_EQ(end_mat_vars[i], start_mat_vars[i], acts=[-gate_var])
        self.encoder.encode_XOR(
            end_mat_vars[j], start_mat_vars[j], start_mat_vars[i], acts=[-gate_var]
        )

    def encode_2q_SWAP(self, start_mat_vars, end_mat_vars, i, j, gate_var):
        """
        Encode act_lit -> (end_mat is obtained from start_mat by applying SWAP(i, j))
        """
        self.encoder.encode_EQ(end_mat_vars[i], start_mat_vars[j], acts=[-gate_var])
        self.encoder.encode_EQ(end_mat_vars[j], start_mat_vars[i], acts=[-gate_var])

    def encode_2q_DCX(self, start_mat_vars, end_mat_vars, i, j, gate_var):
        """
        Encode act_lit -> (end_mat is obtained from start_mat by applying DCX(i, j))
        """
        self.encoder.encode_EQ(end_mat_vars[i], start_mat_vars[j], acts=[-gate_var])
        self.encoder.encode_XOR(
            end_mat_vars[j], start_mat_vars[j], start_mat_vars[i], acts=[-gate_var]
        )

    def _encode_1q_unused(self, layer, start_mat_vars, end_mat_vars, i, used_var):
        """Qubit i is unused."""
        self.encoder.encode_EQ(end_mat_vars[i], start_mat_vars[i], acts=[used_var])

    def _fix_returned_result(self, result: SatProblemResult) -> SatProblemResult:
        return result

    def _check_returned_result(self, result: SatProblemResult):
        checked = result.circuit_with_permutations
        expected = create_unrestricted_circuit(self.final_state)
        ok = StabilizerState(checked).equiv(StabilizerState(expected))
        assert ok
        return ok

    def _process_solution_init_state(self, solution):
        qc = QuantumCircuit(self.nq)
        for i, var in enumerate(self.init_state_vars):
            if solution[var] == 1:
                qc.h(i)
        return qc


def create_unrestricted_circuit(state):
    """Simple function to create qc without any connectivity constraints"""
    first = None
    qc = QuantumCircuit(len(state))
    for i in range(len(state)):
        if state[i] == 0:
            continue
        if first is None:
            first = i
            qc.h(i)
            continue
        qc.cx(first, i)
    return qc
