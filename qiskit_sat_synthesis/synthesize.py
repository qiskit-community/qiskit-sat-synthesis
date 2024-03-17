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
Implements the bottom-up linear search strategy.
"""

import time
import numpy as np

from .sat_problem import SatProblemResult


class SynthesisResult:
    def __init__(self):

        # all solutions
        self.solutions: list[SatProblemResult] = []

        # minimum depth at which a solution has been discovered
        self.min_solution_depth = None

        # total run time
        self.run_time = 0

    def __repr__(self):
        num_solutions = self.num_solutions
        if num_solutions == 0:
            out = f"no solutions in {self.run_time:.2f} s"
        elif num_solutions == 1:
            out = f"found solution at depth {self.min_solution_depth} in {self.run_time:.2f} s"
            res = self.solutions[0]
            if res.num_1q is not None:
                out += f"; num_1q = {res.num_1q}"
            if res.num_2q is not None:
                out += f"; num_2q = {res.num_2q}"
        else:
            out = f"found {self.num_solutions} solutions at depth {self.min_solution_depth} in {self.run_time:.2f} s"
        return out

    @property
    def circuit(self):
        """Returns circuit corresponding to the first satisfiable solution (or None if none)."""
        if len(self.solutions) > 0:
            return self.solutions[0].circuit
        else:
            return None

    @property
    def circuit_with_permutations(self):
        """Returns circuit with permutations corresponding to the first satisfiable solution (or None if none)."""
        if len(self.solutions) > 0:
            return self.solutions[0].circuit_with_permutations
        else:
            return None

    @property
    def is_solved(self):
        """Returns true if at least one satisfying solution has been found."""
        return len(self.solutions) > 0

    @property
    def num_solutions(self):
        """Returns the number of satisfyng solutions."""
        return len(self.solutions)


def synthesize_optimal(
    target_obj,
    create_sat_problem_fn,
    min_depth=0,
    max_depth=np.inf,
    max_unsolved_depths=0,
    max_solutions=1,
    verbosity=1,
) -> SynthesisResult:
    """Uses bottom-up linear search to synthesize optimal object."""

    synthesis_result = SynthesisResult()
    depth = min_depth
    num_unsolved_depths = 0

    start_time = time.time()
    while True:
        if verbosity >= 1:
            print(f"Looking for solution with target depth {depth}")

        # Setup the problem at the current depth
        sat_problem = create_sat_problem_fn(target_obj, depth)

        # Solve
        problem_result = sat_problem.solve()

        # Analyze the result
        if problem_result.is_sat:
            synthesis_result.solutions.append(problem_result)
            synthesis_result.min_solution_depth = depth
            break

        elif problem_result.is_unsat:
            num_unsolved_depths = 0
            depth += 1
            if depth > max_depth:
                break

        else:
            num_unsolved_depths += 1
            if num_unsolved_depths >= max_unsolved_depths:
                break
            depth += 1
            if depth > max_depth:
                break

    if problem_result.is_sat:
        while synthesis_result.num_solutions < max_solutions:
            if verbosity >= 1:
                print(
                    f"Looking for additional solutions (currently found {synthesis_result.num_solutions})"
                )
            problem_result = sat_problem.solve_another()

            if problem_result.is_sat:
                synthesis_result.solutions.append(problem_result)

            else:
                break

    end_time = time.time()
    synthesis_result.run_time = end_time - start_time
    if verbosity >= 1:
        print(f"Synthesis summary: {synthesis_result}")

    return synthesis_result
