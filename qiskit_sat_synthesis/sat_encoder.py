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
The SatEncoder class stores a CNF (conjunctive normal form) encoding of the problem,
and provides a variety of methods to convert more general constraints to the CNF form.
"""

import numpy as np
from qiskit.synthesis.permutation.permutation_utils import _inverse_pattern


class SatEncoder:
    """Stores a CNF encoding of the problem and provides a variety of methods
    to translate more general constraints to the CNF form."""

    def __init__(self, verbosity=0):
        self.MAX_VAR = 0
        self.CNF = []
        self.ASSUMPTIONS = []

        self.eq_hash = dict()
        self.before_hash = dict()

        self.verbosity = verbosity

    def max_var(self):
        return self.MAX_VAR

    def cnf(self):
        return self.CNF

    def new_var(self):
        self.MAX_VAR = self.MAX_VAR + 1
        var = self.MAX_VAR
        return var

    def add_clause(self, c, acts=[]):
        assert isinstance(c, list)
        # for lit in c:
        #     assert(isinstance(lit, (int, np.int32)))

        if self.verbosity >= 4:
            print(f"--adding clause {c + acts}")
        self.CNF.append(c + acts)

    def clear_assumptions(self):
        self.ASSUMPTIONS.clear()

    def add_assumption(self, a):
        self.ASSUMPTIONS.append(a)

    def negate_clause(self, c):
        nc = [-x for x in c]
        return nc

    def print_cnf(self):
        for c in self.CNF:
            print(c)

    def print_stats(self, msg):
        print(f"{msg}: #vars = {self.MAX_VAR}, #clauses = {len(self.CNF)}")

    def write_cnf_to_file(self, output_file_name):
        if self.verbosity >= 1:
            print(
                f"writing problem {output_file_name}: #vars = {self.MAX_VAR}, #clauses = {len(self.CNF)}"
            )
        with open(output_file_name, "w") as write_file:
            header_str = "p cnf " + str(self.MAX_VAR) + " " + str(len(self.CNF)) + "\n"
            write_file.write(header_str)
            for c in self.CNF:
                c_str = ""
                for lit in c:
                    c_str += str(lit) + " "
                c_str += "0\n"
                write_file.write(c_str)
        write_file.close()

    def encode_at_most_one_naive(self, lits, acts=[]):
        if self.verbosity >= 3:
            print(f"Encoding at-most-one constraints on {lits} using naive method:")
        for i in range(len(lits)):
            for j in range(len(lits)):
                if j > i:
                    self.add_clause([-lits[i], -lits[j]], acts)

    def encode_at_most_one_order(self, lits, acts=[]):
        if self.verbosity >= 3:
            print(f"Encoding at-most-one constraints on {lits} using order method:")

        n = len(lits)
        assert n > 3

        ys = [self.new_var() for _ in range(n)]
        self.add_clause([-lits[0], ys[0]], acts)

        for i in range(1, n):
            self.add_clause([ys[i], -ys[i - 1]], acts)
            self.add_clause([ys[i], -lits[i]], acts)
            self.add_clause([-ys[i - 1], -lits[i]], acts)

    def encode_at_most_one(self, lits, acts=[]):
        if len(lits) in [2, 3]:
            self.encode_at_most_one_naive(lits, acts)
        elif len(lits) > 3:
            self.encode_at_most_one_order(lits, acts)

    def encode_exactly_one(self, lits, acts=[]):
        if self.verbosity >= 3:
            print(f"Encoding exactly-one constraints on {lits}:")
        self.add_clause(lits, acts)
        self.encode_at_most_one(lits, acts)

    def encode_unary_counter(self, x, k):
        """Creates partial sums s_{i, j} such that
            ``s_{i, j}=1`` if and only if ``x_0 + ... + x_{i} >= j+1``,
            where i = 0 .. n-1, and j = 0 .. k-1.

        In particular, the unit clause (s_{n-1, j}) would encode "x_0 + ... +x_{n-1} >= j+1",
          for j=0,...,k-1, and
        and the unit clause (!s_{n-1, j}) would encode "x_0 + ... +x_{n-1} <= j",
          for j=0,...,k-1

        Returns s.
        """

        n = len(x)
        s = {}

        for i in range(n):
            for j in range(k):
                s[i, j] = self.new_var()

        for i in range(n):
            for j in range(k):
                if i == 0 and j == 0:
                    self.add_clause([-x[i], s[i, j]])
                    self.add_clause([x[i], -s[i, j]])
                elif i == 0:
                    self.add_clause([-s[i, j]])
                elif j == 0:
                    self.add_clause([-s[i - 1, j], s[i, j]])
                    self.add_clause([-x[i], s[i, j]])
                    self.add_clause([s[i - 1, j], x[i], -s[i, j]])
                else:
                    # s[i, j] = s[i-1, j] OR (s[i-1, j-1] AND x[i])
                    self.add_clause([-s[i - 1, j], s[i, j]])
                    self.add_clause([-s[i - 1, j - 1], -x[i], s[i, j]])
                    self.add_clause([s[i - 1, j], s[i - 1, j - 1], -s[i, j]])
                    self.add_clause([s[i - 1, j], x[i], -s[i, j]])

        return s

    def encode_XOR(self, a, b, c, acts=[]):
        """Encodes activated constraint a XOR b XOR c == 0,
        equivalently c = a XOR b, or c = (a != b)."""
        self.add_clause([-a, b, c], acts)
        self.add_clause([a, -b, c], acts)
        self.add_clause([a, b, -c], acts)
        self.add_clause([-a, -b, -c], acts)

    def encode_XNOR(self, a, b, c, acts=[]):
        """Encodes activated constraint a XOR b XOR c == 1,
        equivalently c = a XNOR b, or c = (a == b)."""
        self.add_clause([a, -b, -c], acts)
        self.add_clause([-a, b, -c], acts)
        self.add_clause([-a, -b, c], acts)
        self.add_clause([a, b, c], acts)

    def encode_general_xor(self, xor_clause, acts=[]):
        """Encodes activated constraints
        (xor_clause[0] XOR ... XOR xor_clause[n-1] == 0).
        """
        xor_size = len(xor_clause)

        if xor_size == 0:
            # tautology, nothing to do
            pass

        elif xor_size == 1:
            # unit clause
            self.add_clause([-xor_clause[0]], acts=acts)

        elif xor_size == 2:
            # equality
            self.encode_EQ(xor_clause[0], xor_clause[1], acts=acts)

        elif xor_size == 3:
            # ternary
            self.encode_XOR(xor_clause[0], xor_clause[1], xor_clause[2], acts=acts)

        else:
            # larger, recursive
            y = self.new_var()
            self.encode_XOR(xor_clause[0], xor_clause[1], y, acts=acts)
            xor_clause_rem = [y]
            for c in xor_clause[2:]:
                xor_clause_rem.append(c)
            self.encode_general_xor(xor_clause_rem, acts=acts)

    # encode activation_lits -> (a=b)
    def encode_EQ(self, a, b, acts=[]):
        """Encode activated constraint a == b
        (equivalently, a XOR b == 0).
        """
        self.add_clause([-a, b], acts)
        self.add_clause([a, -b], acts)

    def encode_smaller(self, control_vars, target_vars, acts=[]):
        for i in range(len(control_vars)):
            for j in range(len(target_vars)):
                if i >= j:
                    self.add_clause([-control_vars[i], -target_vars[j]], acts)

    def encode_OR(self, a, b, c, acts=[]):
        """Encodes c = a | b."""
        self.add_clause([-a, c], acts)
        self.add_clause([-b, c], acts)
        self.add_clause([a, b, -c], acts)

    # encode z = A[0] | ... | A[n-1]
    def encode_EQ_OR(self, A, z):
        for i in range(len(A)):
            self.add_clause([-A[i], z])
        self.add_clause(A + [-z])

    # encode z = A[0] & ... & A[n-1]
    def encode_EQ_AND(self, A, z):
        for i in range(len(A)):
            self.add_clause([A[i], -z])
        self.add_clause([-a for a in A] + [z])

    # encode (A[0] | ... | A[n-1] == (B[0] | ... | B[m-1])
    def encode_OR_EQ_OR(self, A, B):
        for a in A:
            self.add_clause(B + [-a])
        for b in B:
            self.add_clause(A + [-b])

    def encode_AND(self, a, b, c, acts=[]):
        self.add_clause([-c, a], acts)
        self.add_clause([-c, b], acts)
        self.add_clause([c, -a, -b], acts)

    def encode_OR_VEC(self, A, B, C, acts=[]):
        for i in range(len(A)):
            self.encode_OR(A[i], B[i], C[i], acts)

    # encode z = (A[0] & B[0]) | ... | (A[n] & B[n])
    def encode_both_on_together(self, A, B):
        s = str([A, B])

        if s in self.eq_hash.keys():
            return self.eq_hash[s]

        z = self.new_var()
        n = len(A)
        C = [self.new_var() for _ in range(n)]
        [self.encode_AND(A[i], B[i], C[i]) for i in range(n)]
        self.encode_EQ_OR(C, z)

        self.eq_hash[s] = z

        return z

    def encode_vectors_are_equal(self, A, B):
        """Encodes z <-> (A == B) and returns z.

        That is, z <-> ((A[0] = B[0]) & ... & (A[n] = B[n]))
        """
        z = self.new_var()
        n = len(A)
        C = [self.new_var() for _ in range(n)]
        for i in range(n):
            self.encode_XOR(A[i], B[i], -C[i])
        self.encode_EQ_AND(C, z)
        return z

    # encode z = A < B
    def encode_on_before(self, A, B):
        s = str([A, B])

        if s in self.before_hash.keys():
            return self.before_hash[s]

        z = self.new_var()
        n = len(A)
        V = [self.new_var() for _ in range(n)]
        self.add_clause([V[0]])
        for i in range(1, n):
            self.add_clause([-V[i], V[i - 1]])
            self.add_clause([-V[i], -A[i - 1]])
            self.add_clause([-V[i], -B[i - 1]])
            self.add_clause([V[i], -V[i - 1], A[i - 1], B[i - 1]])
        W = [self.new_var() for _ in range(n)]
        for i in range(n):
            self.add_clause([-W[i], A[i]])
            self.add_clause([-W[i], -B[i]])
            self.add_clause([W[i], -A[i], B[i]])
        T = [self.new_var() for _ in range(n)]
        for i in range(n):
            self.encode_AND(V[i], W[i], T[i])
        self.encode_EQ_OR(T, z)

        self.before_hash[s] = z

        return z

    def create_mat_with_new_vars(self, nrows, ncols=None):
        """
        Returns a new nrows x ncols matrix with fresh CNF variables.
        If ncols is None, then returns a square matrix nrows x nrows.
        """
        if ncols is None:
            ncols = nrows
        mat = np.array([[self.new_var() for _ in range(ncols)] for _ in range(nrows)])
        if self.verbosity >= 3:
            print(f"Creating new matrix:\n {mat}")
        return mat

    def encode_mat_is_permutation(self, mat):
        """Given a nq x nq matrix of CNF variables, encode constraints that it represents
        a permutation matrix."""

        nq = len(mat)
        for i in range(nq):
            lits = [mat[i, j] for j in range(nq)]
            self.encode_exactly_one(lits)
        for j in range(nq):
            lits = [mat[i, j] for i in range(nq)]
            self.encode_exactly_one(lits)

    def create_perm_mat_with_new_vars(self, nq):
        """
        Returns a new nq x nq matrix with fresh CNF variables.
        Adds constraints that this is a permutation matrix.
        """
        mat = self.create_mat_with_new_vars(nq)
        self.encode_mat_is_permutation(mat)
        return mat

    def block_solution(self, solution, blocking_vars=None):
        """Adds clause blocking a given solution."""
        considered_vars = (
            list(range(1, len(solution))) if blocking_vars is None else blocking_vars
        )
        blocking_clause = []
        for i in considered_vars:
            if solution[i] == 1:
                blocking_clause.append(-i)
            else:
                blocking_clause.append(i)
        self.add_clause(blocking_clause)

    def encode_at_most_k(self, lits, k):
        if k == 0:
            for lit in lits:
                self.add_clause([-lit])
        elif k == 1:
            self.encode_at_most_one(lits)
        else:
            counter = UnaryCounter(lits, self)
            counter.extend(k + 1)
            self.add_clause([-counter.get_counter_var(k + 1)])

    def get_perm_pattern_from_solution(self, perm_mat, sol):
        """Returns permutation pattern from solution."""
        nq = len(perm_mat)
        perm_pattern = []
        for i in range(nq):
            for j in range(nq):
                lit = perm_mat[i, j]
                val = sol[lit]
                if val == 1:
                    perm_pattern.append(j)
        inverse_perm_pattern = _inverse_pattern(perm_pattern)
        return inverse_perm_pattern


class UnaryCounter:
    """Incremental unary counter."""

    def __init__(self, x, encoder):
        # print(f"init with {x = }, {len(x) = }")
        self.x = x
        self.s = {}
        self.max_k_encoded = 0
        self.encoder = encoder

    def extend(self, k):
        n = len(self.x)
        k1 = self.max_k_encoded
        k2 = k
        s = self.s
        x = self.x
        if k2 <= k1:
            return s

        for i in range(n):
            for j in range(k1, k2):
                s[i, j] = self.encoder.new_var()

        for i in range(n):
            for j in range(k1, k2):
                if i == 0 and j == 0:
                    self.encoder.add_clause([-x[i], s[i, j]])
                    self.encoder.add_clause([x[i], -s[i, j]])
                elif i == 0:
                    self.encoder.add_clause([-s[i, j]])
                elif j == 0:
                    self.encoder.add_clause([-s[i - 1, j], s[i, j]])
                    self.encoder.add_clause([-x[i], s[i, j]])
                    self.encoder.add_clause([s[i - 1, j], x[i], -s[i, j]])
                else:
                    # s[i, j] = s[i-1, j] OR (s[i-1, j-1] AND x[i])
                    self.encoder.add_clause([-s[i - 1, j], s[i, j]])
                    self.encoder.add_clause([-s[i - 1, j - 1], -x[i], s[i, j]])
                    self.encoder.add_clause([s[i - 1, j], s[i - 1, j - 1], -s[i, j]])
                    self.encoder.add_clause([s[i - 1, j], x[i], -s[i, j]])

        self.max_k_encoded = k
        return s

    def get_counter_var(self, k):
        return self.s.get((len(self.x) - 1, k - 1), None)
