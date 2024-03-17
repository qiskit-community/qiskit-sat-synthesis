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

"""The SatSolver class runs the sat-solver on CNF problems."""

from enum import Enum
import z3
import time


class SolverStatus(Enum):
    """Possible outcomes for running the Z3-solver."""

    UNKNOWN = 0
    SAT = 1
    UNSAT = 2
    ERROR = 3


SOLVER_STATUS_TO_STR = {
    SolverStatus.UNKNOWN: "unknown",
    SolverStatus.SAT: "sat",
    SolverStatus.UNSAT: "unsat",
    SolverStatus.ERROR: "error",
}


class SatSolverResult:
    def __init__(self, result, solution, run_time):
        self.result = result
        self.solution = solution
        self.run_time = run_time

    def __repr__(self):
        return (
            SOLVER_STATUS_TO_STR[self.result]
            + " in "
            + str(round(self.run_time, 2))
            + " s"
        )


class SatSolver:
    """
    Runs the Z3-solver on problems encoded into encoder.

    Can be run incrementally as more clauses are added into the encoder.
    """

    def __init__(self, encoder, assumptions_as_unit_clauses=True, verbosity=0):
        self.verbosity = verbosity
        self.sat_solver = z3.Solver()
        self.encoder = encoder
        self.first_unsent = 0
        self.assumptions_as_unit_clauses = assumptions_as_unit_clauses

    def to_z3lit(self, l):
        v = abs(l)
        z3var = z3.Bool("x" + str(v))
        z3lit = z3var if l > 0 else z3.Not(z3var)
        return z3lit

    def add_clause_to_sat_solver(self, c):
        z3lits = [self.to_z3lit(l) for l in c]
        self.sat_solver.add(z3.Or(z3lits))

    def add_cnf_to_sat_solver(self, cnf):
        for c in cnf:
            self.add_clause_to_sat_solver(c)

    def send_new_clauses_to_sat_solver(self):
        CNF = self.encoder.CNF
        self.add_cnf_to_sat_solver(CNF[self.first_unsent :])
        self.first_unsent = len(CNF)

    def solve(self, max_conflicts=None, timeout=None) -> SatSolverResult:
        """
        max_conflicts: maximum number of conflicts (deterministic)
        timeout: time limit in milliseconds, i.e. the value of 5000 corresponds to 5 seconds
          (not deterministic)
        """
        res = SolverStatus.UNKNOWN
        solution = None

        start_time = time.time()

        self.send_new_clauses_to_sat_solver()

        z3_assumptions = [self.to_z3lit(l) for l in self.encoder.ASSUMPTIONS]

        # Add assumptions as unit clauses
        if self.verbosity >= 1:
            print(
                f" => running sat-solver: #vars = {self.encoder.MAX_VAR}, "
                f" #clauses = {len(self.encoder.CNF)}, "
                f" #assumptions = {len(self.encoder.ASSUMPTIONS)}"
            )

        if max_conflicts is not None:
            self.sat_solver.set("max_conflicts", max_conflicts)

        if timeout is not None:
            self.sat_solver.set("timeout", timeout)

        if self.assumptions_as_unit_clauses:
            for a in z3_assumptions:
                self.sat_solver.add(a)
            z3res = self.sat_solver.check()
        else:
            z3res = self.sat_solver.check(z3_assumptions)

        if z3res == z3.sat:
            res = SolverStatus.SAT
            solution = self.solution()

        elif z3res == z3.unsat:
            res = SolverStatus.UNSAT

        end_time = time.time()
        elapsed_time = end_time - start_time
        sat_query_result = SatSolverResult(res, solution, elapsed_time)
        if self.verbosity >= 1:
            print(f" => sat-solver result: {sat_query_result}")
        return sat_query_result

    def solution(self):
        max_var = self.encoder.MAX_VAR
        model = self.sat_solver.model()
        solution = [False] * (max_var + 1)
        for v in range(1, max_var + 1):
            z3var = z3.Bool("x" + str(v))
            if model[z3var]:
                solution[v] = True
        # print(solution)
        return solution
