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
used for creating and encoding general problems into SAT.
"""

from abc import abstractmethod

from itertools import combinations

from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.permutation.permutation_utils import _get_ordered_swap

from .sat_encoder import SatEncoder, UnaryCounter
from .sat_solver import SatSolver, SolverStatus


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


# ToDo: move the constraints code to a separate file


class SatConstraint:
    @abstractmethod
    def encode(self, sat_problem):
        raise NotImplementedError


class Sat1QOnlyAfter2QConstraint(SatConstraint):
    """
    Search-space reduction based on Bravyi's clifford compiler:
    if a 2-qubit layer is followed by a 1-qubit layer,
    then 1q-gates can only be on qubits involved in 2-qubit gates;
    i.e. qubit i is used in 1-qubit layer => qubit i is used in 2-qubit layer.

    Can be used for Clifford's depth and count synthesis based on Bravyj's
    compiler.
    """

    def __init__(self, layer2q_id, layer1q_id):
        self.layer2q_id = layer2q_id
        self.layer1q_id = layer1q_id

    def encode(self, sat_problem):
        layer2q = sat_problem.layers[self.layer2q_id]
        layer1q = sat_problem.layers[self.layer1q_id]
        for i in range(sat_problem.nq):
            sat_problem.encoder.add_clause(
                [-layer1q.q_used_vars[i], layer2q.q_used_vars[i]]
            )


class SatLayerNonEmptyConstraint(SatConstraint):
    """
    Constraint that a given layer is not empty.

    Can be used for 2-qubit layers in Clifford depth and count synthesis, and for
    linear functions depth and count synthesis.
    """

    def __init__(self, layer_id):
        self.layer_id = layer_id

    def encode(self, sat_problem):
        layer = sat_problem.layers[self.layer_id]
        used_vars = list(layer.q_used_vars.values())
        sat_problem.encoder.add_clause(used_vars)


class SatMaxUnique2QLayersConstraint(SatConstraint):
    """
    The constraint restricts the maximum number of unique layers out of the
    given 2-qubit layers.
    """

    def __init__(self, layer_ids, max_unique):
        self.layer_ids = layer_ids
        self.max_unique = max_unique

    def encode(self, sat_problem):
        num_layers = len(self.layer_ids)
        if num_layers <= self.max_unique:
            return

        # encode all (optional) connection vars (if not already)
        for layer_id in self.layer_ids:
            sat_problem.layers[layer_id].encode_connection_vars()

        # for two layers s and t, we will create a variable d[s, t] representing that
        # the two layers are different.
        layers_different_vars = {}
        for s in range(num_layers):
            for t in range(s + 1, num_layers):
                sid = self.layer_ids[s]
                tid = self.layer_ids[t]

                # the layers will be different iff there is at least one lit in lits
                # that evaluates to True
                lits = []

                for i in range(sat_problem.nq):
                    for j in range(i + 1, sat_problem.nq):
                        used_s = sat_problem.layers[sid].get_connection_var(i, j)
                        used_t = sat_problem.layers[tid].get_connection_var(i, j)
                        if used_s is None and used_t is None:
                            pass
                        elif used_s is not None and used_t is None:
                            lits.append(used_s)
                        elif used_s is None and used_t is not None:
                            lits.append(used_t)
                        else:
                            not_eq_var = sat_problem.encoder.new_var()
                            sat_problem.encoder.encode_XOR(used_s, used_t, not_eq_var)
                            lits.append(not_eq_var)

                are_different_lit = sat_problem.encoder.new_var()
                sat_problem.encoder.encode_EQ_OR(lits, are_different_lit)

                layers_different_vars[s, t] = are_different_lit

        # next, create layer_unique_vars, so that layer_unique_vars[t] is True
        # if the layer t is different from the previous layers
        layer_unique_vars = []
        for t in range(1, num_layers):
            is_unique_var = sat_problem.encoder.new_var()
            previous_vars = [layers_different_vars[s, t] for s in range(t)]
            sat_problem.encoder.encode_EQ_AND(previous_vars, is_unique_var)
            layer_unique_vars.append(is_unique_var)

        # create unary counter
        unique_layer_counter = UnaryCounter(layer_unique_vars, sat_problem.encoder)
        unique_layer_counter.extend(self.max_unique)
        counter_var = unique_layer_counter.get_counter_var(self.max_unique)
        if counter_var is not None:
            sat_problem.encoder.add_clause([-counter_var])


class SatCannotPush2QEarlierConstraint(SatConstraint):
    """
    Constraint that a 2-qubit gate in one layer cannot be
    placed instead in another layer.

    Can be called on 2 consecutive 2-qubit layers for Clifford's depth
    and Linear Function's depth synthesis. For Clifford synthesis it is
    also valid to have 1-qubit layers in between, because these can be
    combined with earlier 1-qubit gates.
    """

    def __init__(self, layer1_id, layer2_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id

    def encode(self, sat_problem):
        layer1_id = self.layer1_id
        layer2_id = self.layer2_id

        sat_problem.layers[layer1_id].encode_connection_vars()
        sat_problem.layers[layer2_id].encode_connection_vars()

        for i in range(sat_problem.nq):
            for j in range(i + 1, sat_problem.nq):
                used_in_prev = sat_problem.layers[layer1_id].get_connection_var(i, j)
                used_in_next = sat_problem.layers[layer2_id].get_connection_var(i, j)

                if used_in_prev is None or used_in_next is None:
                    # such gate is not allowed either in the current layer (nothing to do)
                    # or in the previous layer (nothing can be done)
                    continue

                # the constraint is that the following combination is not possible:
                #   (1) we have a gate connecting i and j in the current layer
                #   (2) qubits i and j are free in the previous layer

                q_used_i_prev = sat_problem.layers[layer1_id].q_used_vars[i]
                q_used_j_prev = sat_problem.layers[layer1_id].q_used_vars[j]

                clause = [-used_in_next, q_used_i_prev, q_used_j_prev]
                sat_problem.encoder.add_clause(clause)


class SatLayersIntersectOrOrderedConstraint(SatConstraint):
    """
    The constraint says that two layers over disjoint qubits can be
    assumed to be lexicographically ordered.

    This makes sense for two consecutive layers for Clifford and Linear
    Function synthesis (as the layers can then be permuted). Though it
    only makes sense to apply this for count, since for depth we can
    add a better constraint that such a case does not happen (as the
    layers can then be squished together).

    """

    def __init__(self, layer1_id, layer2_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id

    def encode(self, sat_problem):
        layer1 = sat_problem.layers[self.layer1_id]
        layer2 = sat_problem.layers[self.layer2_id]
        layers_intersect_lit = sat_problem.encoder.encode_both_on_together(
            layer1.q_used_vars, layer2.q_used_vars
        )
        layers_ordered_lit = sat_problem.encoder.encode_on_before(
            layer1.q_used_vars, layer2.q_used_vars
        )
        sat_problem.encoder.add_clause([layers_intersect_lit, layers_ordered_lit])


class SatCannotSimplify2Q2QConstraint(SatConstraint):
    """
    The constraint that pairs of gates in two (consecutive)
    2-qubits layers cannot cancel out/be simplified.

    Right now this includes CX(i, j) CX(i, j).

    Can be used for LinearFunctions depth and count optimal
    synthesis.
    """

    def __init__(self, layer1_id, layer2_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id

    def encode(self, sat_problem):
        """
        Do not allow CX(i, j) CX(i, j).
        More identities (including DCX and SWAP) to be added later.
        """
        layer1 = sat_problem.layers[self.layer1_id]
        layer2 = sat_problem.layers[self.layer2_id]

        if "CX" not in layer1.gates_2q or "CX" not in layer2.gates_2q:
            return

        for i in range(sat_problem.nq):
            for j in range(sat_problem.nq):

                if not layer1.is_connected(i, j) or not layer2.is_connected(i, j):
                    continue

                layer1_cx_var = layer1.q2_vars[i, j, "CX"]
                layer2_cx_var = layer2.q2_vars[i, j, "CX"]

                # The constraint says that the following are not possible:
                #   CX(i, j) is used in layer1, CX(i, j) is used in layer3
                sat_problem.encoder.add_clause([-layer1_cx_var, -layer2_cx_var])


class SatCannotSimplify2Q1Q2QConstraint(SatConstraint):
    """Suppose that we are given 3 consecutive layers "layer1", "layer2", "layer3" where layer1
    and layer3 are 2-qubit layers, and layer2 is a 1-qubit layer. Further, suppose that the only
    2-qubit gate is the downward CX-gate. Suppose that we have gates CX(i, j) SQi(i) SQj(j) CX(i, j),
    where SQi and SQj are some single-qubit gates on qubits i and j respectively (and can only be
    I, SH or HS). Suppose that in addition either SQi or SQj is I. Then this block can be rewritten
    using a single CX-gate (and some 1-qubit gates). Hence, for "default" Clifford synthesis we can
    restrict the search space by stating that such blocks are not present. However, this constraint
    can probably not be used in all other cases.
    """

    def __init__(self, layer1_id, layer2_id, layer3_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id
        self.layer3_id = layer3_id

    def encode(self, sat_problem):
        layer1 = sat_problem.layers[self.layer1_id]
        layer2 = sat_problem.layers[self.layer2_id]
        layer3 = sat_problem.layers[self.layer3_id]

        if "CX" not in layer1.gates_2q or "CX" not in layer3.gates_2q:
            return

        for i in range(sat_problem.nq):
            for j in range(sat_problem.nq):

                if not layer1.is_connected(i, j) or not layer3.is_connected(i, j):
                    continue

                layer1_cx_var = layer1.q2_vars[i, j, "CX"]
                layer3_cx_var = layer3.q2_vars[i, j, "CX"]
                layer2_i_used_var = layer2.q_used_vars[i]
                layer2_j_used_var = layer2.q_used_vars[j]

                # The constraint says that the following are not possible:
                #   (a) CX(i, j) is used in layer1, qubit i is unused in layer2, CX(i, j) is used in layer3
                #   (b) CX(i, j) is used in layer1, qubit j is unused in layer2, CX(i, j) is used in layer3
                sat_problem.encoder.add_clause(
                    [-layer1_cx_var, -layer3_cx_var, layer2_i_used_var]
                )
                sat_problem.encoder.add_clause(
                    [-layer1_cx_var, -layer3_cx_var, layer2_j_used_var]
                )


class SatCommutation2Q2QConstraint(SatConstraint):
    """
    Restricts search space when two consecutive 2-qubit gates are known to be commuting
    and hence can be switched (provided that other relevant qubits are not used).
    Can be used for linear functions count and depth.
    """

    def __init__(self, layer1_id, layer2_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id

    def encode(self, sat_problem):
        layer1 = sat_problem.layers[self.layer1_id]
        layer2 = sat_problem.layers[self.layer2_id]

        # Add the following constraints:
        # When k > j: it's not possible that:
        #   it is possible to have CX(i, j) and CX(i, k) both in layer1 and in layer2
        #   layer1 has CX(i, k) and qubit j is unused
        #   layer3 has CX(i, j) and qubit k is unused
        for i in range(sat_problem.nq):
            target_vars = [
                j
                for j in range(sat_problem.nq)
                if layer1.is_connected(i, j) and layer2.is_connected(i, j)
            ]
            for j, k in combinations(target_vars, 2):
                layer1_ik_var = layer1.q2_vars[i, k, "CX"]
                layer1_j_used_var = layer1.q_used_vars[j]
                layer2_ij_var = layer2.q2_vars[i, j, "CX"]
                layer2_k_used_var = layer2.q_used_vars[k]
                sat_problem.encoder.add_clause(
                    [
                        -layer1_ik_var,
                        layer1_j_used_var,
                        -layer2_ij_var,
                        layer2_k_used_var,
                    ]
                )

        # Add the following constraints:
        # For j > i: it's not possible that:
        #   it is possible to have CX(i, k) and CX(j, k) both in layer1 and in layer2
        #   layer1 has CX(i, k) and qubit j is unused
        #   layer2 has CX(j, k) and qubit i is unused
        for k in range(sat_problem.nq):
            control_vars = [
                i
                for i in range(sat_problem.nq)
                if layer1.is_connected(i, k) and layer2.is_connected(i, k)
            ]
            for i, j in combinations(control_vars, 2):
                layer1_ik_var = layer1.q2_vars[i, k, "CX"]
                layer1_j_used_var = layer1.q_used_vars[j]
                layer2_jk_var = layer2.q2_vars[j, k, "CX"]
                layer2_i_used_var = layer2.q_used_vars[i]
                sat_problem.encoder.add_clause(
                    [
                        -layer1_ik_var,
                        layer1_j_used_var,
                        -layer2_jk_var,
                        layer2_i_used_var,
                    ]
                )


class SatCommutation2Q1Q2QForCountOnlyConstraint(SatConstraint):
    """Generalization of the previous constraints to Clifford's case.
    Currently, can only be used for Clifford count"""

    def __init__(self, layer1_id, layer2_id, layer3_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id
        self.layer3_id = layer3_id

    def encode(self, sat_problem):
        layer1 = sat_problem.layers[self.layer1_id]
        layer2 = sat_problem.layers[self.layer2_id]
        layer3 = sat_problem.layers[self.layer3_id]

        # Add the following constraints:
        # When k > j: it's not possible that:
        #   layer1 has CX(i, k),
        #   layer3 has CX(i, j), and
        #   qubit i is not used in layer2.
        for i in range(sat_problem.nq):
            target_vars = [
                j
                for j in range(sat_problem.nq)
                if layer1.is_connected(i, j) and layer3.is_connected(i, j)
            ]
            for t1 in range(len(target_vars)):
                for t2 in range(t1 + 1, len(target_vars)):
                    j = target_vars[t1]
                    k = target_vars[t2]
                    layer1_ik_var = layer1.q2_vars[i, k, "CX"]
                    layer3_ij_var = layer3.q2_vars[i, j, "CX"]
                    layer2_i_used_var = layer2.q_used_vars[i]
                    sat_problem.encoder.add_clause(
                        [-layer1_ik_var, -layer3_ij_var, layer2_i_used_var]
                    )

        # Add the following constraints:
        # For j > i: it's not possible that:
        #   layer1 has CX(i, k),
        #   layer3 has CX(j, k), and
        #   qubit k is not used in layer2.
        for k in range(sat_problem.nq):
            control_vars = [
                i
                for i in range(sat_problem.nq)
                if layer1.is_connected(i, k) and layer3.is_connected(i, k)
            ]
            for c1 in range(len(control_vars)):
                for c2 in range(c1 + 1, len(control_vars)):
                    i = control_vars[c1]
                    j = control_vars[c2]
                    layer1_ik_var = layer1.q2_vars[i, k, "CX"]
                    layer3_jk_var = layer3.q2_vars[j, k, "CX"]
                    layer2_k_used_var = layer2.q_used_vars[k]
                    sat_problem.encoder.add_clause(
                        [-layer1_ik_var, -layer3_jk_var, layer2_k_used_var]
                    )


class SatCommutation2Q1Q2QConstraint(SatConstraint):
    """Generalization of the previous constraints to Clifford's case."""

    def __init__(self, layer1_id, layer2_id, layer3_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id
        self.layer3_id = layer3_id

    def encode(self, sat_problem):
        layer1 = sat_problem.layers[self.layer1_id]
        layer2 = sat_problem.layers[self.layer2_id]
        layer3 = sat_problem.layers[self.layer3_id]

        # Add the following constraints:
        # When k > j: it's not possible that:
        #   layer1 has CX(i, k),
        #   layer3 has CX(i, j), and
        #   qubit i is not used in layer2.
        for i in range(sat_problem.nq):
            target_vars = [
                j
                for j in range(sat_problem.nq)
                if layer1.is_connected(i, j) and layer3.is_connected(i, j)
            ]
            for j, k in combinations(target_vars, 2):
                layer1_ik_var = layer1.q2_vars[i, k, "CX"]
                layer1_j_used_var = layer1.q_used_vars[j]
                layer2_i_used_var = layer2.q_used_vars[i]
                layer3_ij_var = layer3.q2_vars[i, j, "CX"]
                layer3_k_used_var = layer3.q_used_vars[k]
                sat_problem.encoder.add_clause(
                    [
                        -layer1_ik_var,
                        layer1_j_used_var,
                        layer2_i_used_var,
                        -layer3_ij_var,
                        layer3_k_used_var,
                    ]
                )

        # Add the following constraints:
        # For j > i: it's not possible that:
        #   layer1 has CX(i, k),
        #   layer3 has CX(j, k), and
        #   qubit k is not used in layer2.
        for k in range(sat_problem.nq):
            control_vars = [
                i
                for i in range(sat_problem.nq)
                if layer1.is_connected(i, k) and layer3.is_connected(i, k)
            ]
            for i, j in combinations(control_vars, 2):
                layer1_ik_var = layer1.q2_vars[i, k, "CX"]
                layer1_j_used_var = layer1.q_used_vars[j]
                layer2_k_used_var = layer2.q_used_vars[k]
                layer3_jk_var = layer3.q2_vars[j, k, "CX"]
                layer3_i_used_var = layer3.q_used_vars[i]
                sat_problem.encoder.add_clause(
                    [
                        -layer1_ik_var,
                        layer1_j_used_var,
                        layer2_k_used_var,
                        -layer3_jk_var,
                        layer3_i_used_var,
                    ]
                )


class SatNonOverlapping2qConstraint(SatConstraint):
    """
    Constrains that a given layer has no "overlapping" 2-qubit gates.
    Currently, this constraint only makes sense for LNN connectivity.
    More precisely, if the first two-qubit gate is over qubits {a, b} with a<b,
    and the second qubit gate is over qubits {c, d} with c<d,
    then the gates overlap iff the qubits are ordered as: {a, c, d, b},
    {c, a, b, d}, {a, c, b, d}, {c, a, d, b}.
    """

    def __init__(self, layer_id):
        self.layer_id = layer_id

    def encode(self, sat_problem):
        def block(a, b, c, d):
            if (used_ab := layer.get_connection_var(a, b)) is None:
                return
            if (used_cd := layer.get_connection_var(c, d)) is None:
                return
            sat_problem.encoder.add_clause([-used_ab, -used_cd])

        layer = sat_problem.layers[self.layer_id]
        layer.encode_connection_vars()

        for a, c, b, d in combinations(range(layer.nq), 4):
            block(a, b, c, d)
        for a, c, d, b in combinations(range(layer.nq), 4):
            block(a, b, c, d)
        for c, a, b, d in combinations(range(layer.nq), 4):
            block(a, b, c, d)
        for c, a, d, b in combinations(range(layer.nq), 4):
            block(a, b, c, d)


class SatLayer:
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
            gate_nq = self.sat_problem.num_qubits_in_gate(gate)
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

    def get_connection_var(self, i, j):
        """Returns connection_vars[i, j] when created."""
        if not self.connection_vars:
            return None
        return self.connection_vars.get((i, j), None)

    def encode_connection_vars(self):
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

    def process_solution(self, solution):
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

    def encode_coupling_map_constraints(self):
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

    def encode_layer_constraints(self):

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
        self.encode_coupling_map_constraints()


class SatProblem:
    """
    Base class for the semi-abstract sat problems.

    Child classes are required to implement the abstracted methods, and may
    reimplement other methods.
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

    def fix_1q_gate_in_layer(self, layer_id, i, g):
        """Add the constraint that we have the 1-q gate g on qubit i in the given layer."""
        layer: SatLayer = self.layers[layer_id]
        assert (
            g in layer.gates_1q
        ), f"Gate {g} is not in the list of gates allowed in layer {layer_id}."
        self.encoder.add_clause([layer.q1_vars[i, g]])

    def exclude_1q_gate_in_layer(self, layer_id, i, g):
        """Add the constraint that we do not have the 1-q gate g on qubit i in the given layer."""
        layer: SatLayer = self.layers[layer_id]
        assert (
            g in layer.gates_1q
        ), f"Gate {g} is not in the list of gates allowed in layer {layer_id}."
        self.encoder.add_clause([-layer.q1_vars[i, g]])

    def fix_2q_gate_in_layer(self, layer_id, i, j, g):
        """Add the constraint that we have the 2-q gate g on qubits i,j in the given layer."""
        layer: SatLayer = self.layers[layer_id]
        assert (
            g in layer.gates_2q
        ), f"Gate {g} is not in the list of gates allowed in layer {layer_id}."
        assert layer.is_connected(
            i, j
        ), f"Qubits {i} and {j} are not connected in layer {layer_id}."
        self.encoder.add_clause([layer.q2_vars[i, j, g]])

    def exclude_2q_gate_in_layer(self, layer_id, i, j, g):
        """Add the constraint that we cannot have the 2-q gate g on qubits i,j in the given layer."""
        layer: SatLayer = self.layers[layer_id]
        assert (
            g in layer.gates_2q
        ), f"Gate {g} is not in the list of gates allowed in layer {layer_id}."
        assert layer.is_connected(
            i, j
        ), f"Qubits {i} and {j} are not connected in layer {layer_id}."
        self.encoder.add_clause([-layer.q2_vars[i, j, g]])

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

            init_state_circuit = self.process_solution_init_state(solver_solution)
            if init_state_circuit is not None:
                qc.append(init_state_circuit, range(self.nq), range(nc))
                qc.barrier()

            for layer in self.layers:
                layer_circuit, layer_1q, layer_2q = layer.process_solution(
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

    def add_layer(
        self,
        gates,
        coupling_maps,
        full_2q=False,
        max_num_1q_gates=None,
        max_num_2q_gates=None,
    ) -> int:
        layer = SatLayer(
            self,
            gates=gates,
            coupling_maps=coupling_maps,
            full_2q=full_2q,
            max_num_1q_gates=max_num_1q_gates,
            max_num_2q_gates=max_num_2q_gates,
        )
        self.layers.append(layer)
        return len(self.layers) - 1

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
                result = self.optimize_num_2q_gates(max_num_1q_gates=None)
            if self.optimize_1q_gates:
                num_2q = result.num_2q
                result = self.optimize_num_1q_gates(max_num_2q_gates=num_2q)

            result = self.fix_returned_result(result)
            if self.check_solutions:
                if self.verbosity >= 1:
                    print(f"Checking solution")

                ok = self.check_returned_result(result)
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
            self.last_result.solver_solution, vars=self.problem_vars
        )
        sat_query_result = self.solver.solve(
            max_conflicts=self.max_conflicts_per_call, timeout=self.timeout_per_call
        )
        result = self._extract_sat_problem_result(sat_query_result)
        self.last_result = result
        if self.verbosity >= 1:
            print(f" => additional synthesis problem result: {self.last_result}")
        if result.is_sat:
            result = self.fix_returned_result(result)
            if self.check_solutions:
                if self.verbosity >= 1:
                    print(f"Checking solution")
                ok = self.check_returned_result(result)
                if self.verbosity >= 1:
                    print(f" => solution is correct: {ok}")
            if self.print_solutions:
                print(f"Printing solution")
                print(result.circuit)
        return result

    def optimize_num_2q_gates(self, max_num_1q_gates):
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

    def optimize_num_1q_gates(self, max_num_2q_gates):
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
                        self.encode_2q_gate(
                            layer, start_state_vars, end_state_vars, i, j, g
                        )

        # For each 1-qubit CNF variable [i, g], encode the update state constraint on
        # qubit i.
        for i in range(self.nq):
            for g in layer.gates_1q:
                self.encode_1q_gate(layer, start_state_vars, end_state_vars, i, g)

        # For each qubit i, if it is not used in any gate, then the
        # state is not changed.
        for i in range(self.nq):
            self.encode_1q_unused(
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
            layer.encode_layer_constraints()
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
            constraint.encode(self)

    # ADDING SPECIAL CONSTRAINTS

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

    # METHODS THAT DERIVED CLASS MUST IMPLEMENT

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
    def encode_1q_gate(self, layer, start_state_vars, end_state_vars, i, g):
        raise NotImplementedError

    @abstractmethod
    def encode_2q_gate(self, layer, start_state_vars, end_state_vars, i, j, g):
        raise NotImplementedError

    @abstractmethod
    def encode_1q_unused(self, layer, start_state_vars, end_state_vars, i, used_var):
        raise NotImplementedError

    @abstractmethod
    def fix_returned_result(self, result: SatProblemResult) -> SatProblemResult:
        raise NotImplementedError

    @abstractmethod
    def check_returned_result(self, result: SatProblemResult):
        raise NotImplementedError

    def process_solution_init_state(self, solution):
        return None

    def num_qubits_in_gate(self, gate):
        """Returns number of qubits used for this gate (supported values are 1 in 2)"""
        if gate in _GATES_1Q:
            return 1
        elif gate in _GATES_2Q:
            return 2
        else:
            raise Exception(f"Gate {gate} is not in the lists of known gates.")
