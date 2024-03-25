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

"""Additional constraints restricting one or several problem layers."""

from abc import abstractmethod

from itertools import combinations

from .sat_encoder import UnaryCounter


class SatProblemConstraint:
    @abstractmethod
    def _encode(self, sat_problem):
        raise NotImplementedError


class Sat1QOnlyAfter2QConstraint(SatProblemConstraint):
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

    def _encode(self, sat_problem):
        layer2q = sat_problem.layers[self.layer2q_id]
        layer1q = sat_problem.layers[self.layer1q_id]
        for i in range(sat_problem.nq):
            sat_problem.encoder.add_clause(
                [-layer1q.q_used_vars[i], layer2q.q_used_vars[i]]
            )


class SatLayerNonEmptyConstraint(SatProblemConstraint):
    """
    Constraint that a given layer is not empty.

    Can be used for 2-qubit layers in Clifford depth and count synthesis, and for
    linear functions depth and count synthesis.
    """

    def __init__(self, layer_id):
        self.layer_id = layer_id

    def _encode(self, sat_problem):
        layer = sat_problem.layers[self.layer_id]
        used_vars = list(layer.q_used_vars.values())
        sat_problem.encoder.add_clause(used_vars)


class SatMaxUnique2QLayersConstraint(SatProblemConstraint):
    """
    The constraint restricts the maximum number of unique layers out of the
    given 2-qubit layers.
    """

    def __init__(self, layer_ids, max_unique):
        self.layer_ids = layer_ids
        self.max_unique = max_unique

    def _encode(self, sat_problem):
        num_layers = len(self.layer_ids)
        if num_layers <= self.max_unique:
            return

        # encode all (optional) connection vars (if not already)
        for layer_id in self.layer_ids:
            sat_problem.layers[layer_id]._encode_connection_vars()

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
                        used_s = sat_problem.layers[sid]._get_connection_var(i, j)
                        used_t = sat_problem.layers[tid]._get_connection_var(i, j)
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


class SatCannotPush2QEarlierConstraint(SatProblemConstraint):
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

    def _encode(self, sat_problem):
        layer1_id = self.layer1_id
        layer2_id = self.layer2_id

        sat_problem.layers[layer1_id]._encode_connection_vars()
        sat_problem.layers[layer2_id]._encode_connection_vars()

        for i in range(sat_problem.nq):
            for j in range(i + 1, sat_problem.nq):
                used_in_prev = sat_problem.layers[layer1_id]._get_connection_var(i, j)
                used_in_next = sat_problem.layers[layer2_id]._get_connection_var(i, j)

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


class SatLayersIntersectOrOrderedConstraint(SatProblemConstraint):
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

    def _encode(self, sat_problem):
        layer1 = sat_problem.layers[self.layer1_id]
        layer2 = sat_problem.layers[self.layer2_id]
        layers_intersect_lit = sat_problem.encoder.encode_both_on_together(
            layer1.q_used_vars, layer2.q_used_vars
        )
        layers_ordered_lit = sat_problem.encoder.encode_on_before(
            layer1.q_used_vars, layer2.q_used_vars
        )
        sat_problem.encoder.add_clause([layers_intersect_lit, layers_ordered_lit])


class SatCannotSimplify2Q2QConstraint(SatProblemConstraint):
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

    def _encode(self, sat_problem):
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


class SatCannotSimplify2Q1Q2QConstraint(SatProblemConstraint):
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

    def _encode(self, sat_problem):
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


class SatCommutation2Q2QConstraint(SatProblemConstraint):
    """
    Restricts search space when two consecutive 2-qubit gates are known to be commuting
    and hence can be switched (provided that other relevant qubits are not used).
    Can be used for linear functions count and depth.
    """

    def __init__(self, layer1_id, layer2_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id

    def _encode(self, sat_problem):
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


class SatCommutation2Q1Q2QForCountOnlyConstraint(SatProblemConstraint):
    """Generalization of the previous constraints to Clifford's case.
    Currently, can only be used for Clifford count"""

    def __init__(self, layer1_id, layer2_id, layer3_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id
        self.layer3_id = layer3_id

    def _encode(self, sat_problem):
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


class SatCommutation2Q1Q2QConstraint(SatProblemConstraint):
    """Generalization of the previous constraints to Clifford's case."""

    def __init__(self, layer1_id, layer2_id, layer3_id):
        self.layer1_id = layer1_id
        self.layer2_id = layer2_id
        self.layer3_id = layer3_id

    def _encode(self, sat_problem):
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


class SatNonOverlapping2qConstraint(SatProblemConstraint):
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

    def _encode(self, sat_problem):
        def block(a, b, c, d):
            if (used_ab := layer._get_connection_var(a, b)) is None:
                return
            if (used_cd := layer._get_connection_var(c, d)) is None:
                return
            sat_problem.encoder.add_clause([-used_ab, -used_cd])

        layer = sat_problem.layers[self.layer_id]
        layer._encode_connection_vars()

        for a, c, b, d in combinations(range(layer.nq), 4):
            block(a, b, c, d)
        for a, c, d, b in combinations(range(layer.nq), 4):
            block(a, b, c, d)
        for c, a, b, d in combinations(range(layer.nq), 4):
            block(a, b, c, d)
        for c, a, d, b in combinations(range(layer.nq), 4):
            block(a, b, c, d)
