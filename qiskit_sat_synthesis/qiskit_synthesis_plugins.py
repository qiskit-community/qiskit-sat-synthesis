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
Synthesis plugins available for qiskit transpiler.

* PermutationGate:
  - SatSynthesisPermutationDepth (key: "sat_depth"): depth-optimal synthesis
  - SatSynthesisPermutationCount (key: "sat_count"): count-optimal synthesis

* LinearFunction:
  - SatSynthesisLinearFunctionDepth (key: "sat_depth"): depth-optimal synthesis
  - SatSynthesisLinearFunctionCount (key: "sat_count"): count-optimal synthesis

* Clifford:
  - SatSynthesisCliffordDepth (key: "sat_depth"): depth-optimal synthesis
  - SatSynthesisCliffordCount ("sat_count"): count-optimal synthesis

Notes:

Each synthesis plugin implements the function

run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options).

The argument `options` is a free-form dictionary allowing to pass additional options
to the synthesis method.

A plugin can output `None` if synthesis does not succeed.

The synthesized circuit is supposed to adhere to connectivity constraints when both
`coupling_map` and `qubits` are specified.
"""

import numpy as np

from qiskit.circuit.library.generalized_gates import PermutationGate, LinearFunction
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap

from qiskit.transpiler.passes.synthesis.hls_plugins import (
    HighLevelSynthesisPlugin,
)

from .synthesize_permutation import (
    synthesize_permutation_depth,
    synthesize_permutation_count,
)
from .synthesize_linear import (
    synthesize_linear_depth,
    synthesize_linear_count,
)
from .synthesize_clifford import (
    synthesize_clifford_depth,
    synthesize_clifford_count,
)

from .utils import perm_matrix


class SatSynthesisPermutationDepth(HighLevelSynthesisPlugin):
    """Synthesis plugin that uses SAT to find a depth-optimal
    SWAP-circuit implementing a given permutation. The name of
    the plugin is ``permutation.sat_depth``.

    In more detail, this plugin is used to synthesize Qiskit objects
    of type `PermutationGate`. When synthesis succeeds, the plugin
    outputs a quantum circuit consisting only of swap gates.

    The plugin supports the following additional options:

    * verbosity (int): output verbosity (0 = do not show any output)
    * min_depth2q (int): minimum depth to consider
    * max_depth2q (int): maximum depth to consider
    * allow_layout_permutation (bool): synthesize up to a layout permutation
    * allow_final_permutation (bool): synthesize up to a final permutation
    * max_2q_per_layer (int | None): maximum number of gates per layer
    * max_num_2q_gates (int | None): maximum total number of gates
    * max_num_unique_layers (int | None): maximum number of unique layers
    * optimize_2q_gates (bool): after finding the optimal depth, additionally minimize
        the total number of gates for this depth
    * restrict_search_space (bool): use optimizations for restricting search space
    * max_solutions (int): maximum number of solutions to discover (different solutions
        can be displayed, but only the first one is returned)
    * check_solutions (bool): self-check that the solution is correct (for debugging)
    * print_solutions (bool): print solutions
    * timeout_per_call (int): time limit for sat solver query in milliseconds
    * max_conflicts_per_call (int): max conflicts for sat solver query
    * max_unsolved_depths (int): maximum number of depths without sat/unsat answer

    """

    def run(
        self, high_level_object, coupling_map=None, target=None, qubits=None, **options
    ):
        """Run synthesis for the given `PermutationGate`."""

        verbosity = options.get("verbosity", 0)
        min_depth2q = options.get("min_depth", 0)
        max_depth2q = options.get("max_depth2q", np.inf)
        allow_layout_permutation = options.get("allow_layout_permutation", False)
        allow_final_permutation = options.get("allow_final_permutation", False)
        max_2q_per_layer = options.get("max_2q_per_layer", None)
        max_num_2q_gates = options.get("max_num_2q_gates", None)
        max_num_unique_layers = options.get("max_num_2q_gates", None)
        optimize_2q_gates = options.get("optimize_2q_gates", False)
        restrict_search_space = options.get("restrict_search_space", True)
        max_solutions = options.get("max_solutions", 1)
        check_solutions = options.get("check_solutions", True)
        print_solutions = options.get("print_solutions", False)
        timeout_per_call = options.get("timeout_per_call", None)
        max_conflicts_per_call = options.get("max_conflicts_per_call", None)
        max_unsolved_depths = options.get("max_unsolved_depths", 0)
        if not isinstance(high_level_object, PermutationGate):
            raise Exception(
                "SatSynthesisPermutationDepth plugin only works with objects of type PermutationGate."
            )

        pattern = high_level_object.pattern
        mat = perm_matrix(pattern)

        if coupling_map is None or qubits is None:
            used_coupling_map = CouplingMap.from_full(len(pattern))
        else:
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        result = synthesize_permutation_depth(
            mat=mat,
            coupling_map=used_coupling_map,
            max_depth2q=max_depth2q,
            min_depth2q=min_depth2q,
            allow_layout_permutation=allow_layout_permutation,
            allow_final_permutation=allow_final_permutation,
            max_2q_per_layer=max_2q_per_layer,
            max_num_2q_gates=max_num_2q_gates,
            max_num_unique_layers=max_num_unique_layers,
            optimize_2q_gates=optimize_2q_gates,
            check_solutions=check_solutions,
            print_solutions=print_solutions,
            max_solutions=max_solutions,
            restrict_search_space=restrict_search_space,
            timeout_per_call=timeout_per_call,
            max_conflicts_per_call=max_conflicts_per_call,
            max_unsolved_depths=max_unsolved_depths,
            verbosity=verbosity,
        )

        return result.circuit


class SatSynthesisPermutationCount(HighLevelSynthesisPlugin):
    """Synthesis plugin that uses SAT to find a count-optimal
    SWAP-circuit implementing a given permutation. The name of
    the plugin is ``permutation.sat_count``.

    In more detail, this plugin is used to synthesize Qiskit objects
    of type `PermutationGate`. When synthesis succeeds, the plugin
    outputs a quantum circuit consisting only of swap gates.

    The plugin supports the following additional options:

    * verbosity (int): output verbosity (0 = do not show any output)
    * min_depth2q (int): minimum depth to consider
    * max_depth2q (int): maximum depth to consider
    * allow_layout_permutation (bool): synthesize up to a layout permutation
    * allow_final_permutation (bool): synthesize up to a final permutation
    * restrict_search_space (bool): use optimizations for restricting search space
    * max_solutions (int): maximum number of solutions to discover (different solutions
        can be displayed, but only the first one is returned)
    * check_solutions (bool): self-check that the solution is correct (for debugging)
    * print_solutions (bool): print solutions
    * timeout_per_call (int): time limit for sat solver query in milliseconds
    * max_conflicts_per_call (int): max conflicts for sat solver query
    * max_unsolved_depths (int): maximum number of depths without sat/unsat answer

    """

    def run(
        self, high_level_object, coupling_map=None, target=None, qubits=None, **options
    ):
        """Run synthesis for the given `PermutationGate`."""

        verbosity = options.get("verbosity", 0)
        min_depth2q = options.get("min_depth", 0)
        max_depth2q = options.get("max_depth2q", np.inf)
        allow_layout_permutation = options.get("allow_layout_permutation", False)
        allow_final_permutation = options.get("allow_final_permutation", False)
        restrict_search_space = options.get("restrict_search_space", True)
        max_solutions = options.get("max_solutions", 1)
        check_solutions = options.get("check_solutions", True)
        print_solutions = options.get("print_solutions", False)
        timeout_per_call = options.get("timeout_per_call", None)
        max_conflicts_per_call = options.get("max_conflicts_per_call", None)
        max_unsolved_depths = options.get("max_unsolved_depths", 0)

        if not isinstance(high_level_object, PermutationGate):
            raise Exception(
                "SatSynthesisPermutationDepth plugin only works with objects of type PermutationGate."
            )

        pattern = high_level_object.pattern
        mat = perm_matrix(pattern)

        if coupling_map is None or qubits is None:
            used_coupling_map = CouplingMap.from_full(len(pattern))
        else:
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        result = synthesize_permutation_count(
            mat=mat,
            coupling_map=used_coupling_map,
            max_depth2q=max_depth2q,
            min_depth2q=min_depth2q,
            allow_layout_permutation=allow_layout_permutation,
            allow_final_permutation=allow_final_permutation,
            check_solutions=check_solutions,
            print_solutions=print_solutions,
            max_solutions=max_solutions,
            restrict_search_space=restrict_search_space,
            timeout_per_call=timeout_per_call,
            max_conflicts_per_call=max_conflicts_per_call,
            max_unsolved_depths=max_unsolved_depths,
            verbosity=verbosity,
        )

        return result.circuit


class SatSynthesisLinearFunctionDepth(HighLevelSynthesisPlugin):
    """Synthesis plugin that uses SAT to find a depth-optimal
    CX-circuit implementing a given linear function. The name of
    the plugin is ``linear_function.sat_depth``.

    In more detail, this plugin is used to synthesize Qiskit objects
    of type `LinearFunction`. When synthesis succeeds, the plugin
    outputs a quantum circuit consisting only of CX gates.

    The plugin supports the following additional options:

    * verbosity (int): output verbosity (0 = do not show any output)
    * min_depth2q (int): minimum depth to consider
    * max_depth2q (int): maximum depth to consider
    * allow_layout_permutation (bool): synthesize up to a layout permutation
    * allow_final_permutation (bool): synthesize up to a final permutation
    * max_2q_per_layer (int | None): maximum number of gates per layer
    * max_num_2q_gates (int | None): maximum total number of gates
    * max_num_unique_layers (int | None): maximum number of unique layers
    * optimize_2q_gates (bool): after finding the optimal depth, additionally minimize
        the total number of gates for this depth
    * restrict_search_space (bool): use optimizations for restricting search space
    * max_solutions (int): maximum number of solutions to discover (different solutions
        can be displayed, but only the first one is returned)
    * check_solutions (bool): self-check that the solution is correct (for debugging)
    * print_solutions (bool): print solutions
    * timeout_per_call (int): time limit for sat solver query in milliseconds
    * max_conflicts_per_call (int): max conflicts for sat solver query
    * max_unsolved_depths (int): maximum number of depths without sat/unsat answer

    """

    def run(
        self, high_level_object, coupling_map=None, target=None, qubits=None, **options
    ):
        """Run synthesis for the given `LinearFunction`."""
        verbosity = options.get("verbosity", 0)
        min_depth2q = options.get("min_depth", 0)
        max_depth2q = options.get("max_depth2q", np.inf)
        allow_layout_permutation = options.get("allow_layout_permutation", False)
        allow_final_permutation = options.get("allow_final_permutation", False)
        max_2q_per_layer = options.get("max_2q_per_layer", None)
        max_num_2q_gates = options.get("max_num_2q_gates", None)
        max_num_unique_layers = options.get("max_num_2q_gates", None)
        optimize_2q_gates = options.get("optimize_2q_gates", False)
        restrict_search_space = options.get("restrict_search_space", True)
        max_solutions = options.get("max_solutions", 1)
        check_solutions = options.get("check_solutions", True)
        print_solutions = options.get("print_solutions", False)
        timeout_per_call = options.get("timeout_per_call", None)
        max_conflicts_per_call = options.get("max_conflicts_per_call", None)
        max_unsolved_depths = options.get("max_unsolved_depths", 0)

        if not isinstance(high_level_object, LinearFunction):
            raise Exception(
                "SatSynthesisLinearFunctionDepth plugin only works with objects of type LinearFunction."
            )

        mat = high_level_object.linear

        if coupling_map is None or qubits is None:
            used_coupling_map = CouplingMap.from_full(len(mat))
        else:
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        result = synthesize_linear_depth(
            mat=mat,
            coupling_map=used_coupling_map,
            max_depth2q=max_depth2q,
            min_depth2q=min_depth2q,
            allow_layout_permutation=allow_layout_permutation,
            allow_final_permutation=allow_final_permutation,
            max_2q_per_layer=max_2q_per_layer,
            max_num_2q_gates=max_num_2q_gates,
            max_num_unique_layers=max_num_unique_layers,
            optimize_2q_gates=optimize_2q_gates,
            check_solutions=check_solutions,
            print_solutions=print_solutions,
            max_solutions=max_solutions,
            restrict_search_space=restrict_search_space,
            timeout_per_call=timeout_per_call,
            max_conflicts_per_call=max_conflicts_per_call,
            max_unsolved_depths=max_unsolved_depths,
            verbosity=verbosity,
        )

        return result.circuit


class SatSynthesisLinearFunctionCount(HighLevelSynthesisPlugin):
    """Synthesis plugin that uses SAT to find a count-optimal
    CX-circuit implementing a given linear function. The name of
    the plugin is ``linear_function.sat_count``.

    In more detail, this plugin is used to synthesize Qiskit objects
    of type `LinearFunction`. When synthesis succeeds, the plugin
    outputs a quantum circuit consisting only of CX gates.

    The plugin supports the following additional options:

    * verbosity (int): output verbosity (0 = do not show any output)
    * min_depth2q (int): minimum depth to consider
    * max_depth2q (int): maximum depth to consider
    * allow_layout_permutation (bool): synthesize up to a layout permutation
    * allow_final_permutation (bool): synthesize up to a final permutation
    * restrict_search_space (bool): use optimizations for restricting search space
    * max_solutions (int): maximum number of solutions to discover (different solutions
        can be displayed, but only the first one is returned)
    * check_solutions (bool): self-check that the solution is correct (for debugging)
    * print_solutions (bool): print solutions
    * timeout_per_call (int): time limit for sat solver query in milliseconds
    * max_conflicts_per_call (int): max conflicts for sat solver query
    * max_unsolved_depths (int): maximum number of depths without sat/unsat answer

    """

    def run(
        self, high_level_object, coupling_map=None, target=None, qubits=None, **options
    ):
        """Run synthesis for the given `LinearFunction`."""

        verbosity = options.get("verbosity", 0)
        min_depth2q = options.get("min_depth", 0)
        max_depth2q = options.get("max_depth2q", np.inf)
        allow_layout_permutation = options.get("allow_layout_permutation", False)
        allow_final_permutation = options.get("allow_final_permutation", False)
        restrict_search_space = options.get("restrict_search_space", True)
        max_solutions = options.get("max_solutions", 1)
        check_solutions = options.get("check_solutions", True)
        print_solutions = options.get("print_solutions", False)
        timeout_per_call = options.get("timeout_per_call", None)
        max_conflicts_per_call = options.get("max_conflicts_per_call", None)
        max_unsolved_depths = options.get("max_unsolved_depths", 0)

        if not isinstance(high_level_object, LinearFunction):
            raise Exception(
                "SatSynthesisPermutationDepth plugin only works with objects of type LinearFunction."
            )

        mat = high_level_object.linear

        if coupling_map is None or qubits is None:
            used_coupling_map = CouplingMap.from_full(len(mat))
        else:
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        result = synthesize_linear_count(
            mat=mat,
            coupling_map=used_coupling_map,
            max_depth2q=max_depth2q,
            min_depth2q=min_depth2q,
            allow_layout_permutation=allow_layout_permutation,
            allow_final_permutation=allow_final_permutation,
            check_solutions=check_solutions,
            print_solutions=print_solutions,
            max_solutions=max_solutions,
            restrict_search_space=restrict_search_space,
            timeout_per_call=timeout_per_call,
            max_conflicts_per_call=max_conflicts_per_call,
            max_unsolved_depths=max_unsolved_depths,
            verbosity=verbosity,
        )

        return result.circuit


class SatSynthesisCliffordDepth(HighLevelSynthesisPlugin):
    """Synthesis plugin that uses SAT to find a depth-optimal
    Clifford circuit implementing a given Clifford. The name of
    the plugin is ``clifford.sat_depth``.

    In more detail, this plugin is used to synthesize Qiskit objects
    of type `Clifford`. When synthesis succeeds, the plugin
    outputs a quantum circuit consisting of CX, S and H gates.

    The plugin supports the following additional options:

    * verbosity (int): output verbosity (0 = do not show any output)
    * state_preparation_mode (bool): synthesize state rather than the full clifford
    * min_depth2q (int): minimum depth to consider
    * max_depth2q (int): maximum depth to consider
    * allow_layout_permutation (bool): synthesize up to a layout permutation
    * allow_final_permutation (bool): synthesize up to a final permutation
    * max_2q_per_layer (int | None): maximum number of 2q-gates per layer
    * max_1q_per_layer (int | None): maximum number of 1q-gates per layer
    * max_num_2q_gates (int | None): maximum total number of 2a-gates
    * max_num_1q_gates (int | None): maximum total number of 1q-gates
    * max_num_unique_layers (int | None): maximum number of unique layers
    * optimize_2q_gates (bool): after finding the optimal depth, additionally minimize
        the total number of 2-qubit gates for this depth
    * optimize_1q_gates (bool): after minimizing the 2q-depth and the number of 2q-gates,
        additionally minimizes the number of 1q-gates
    * restrict_search_space (bool): use optimizations for restricting search space
    * max_solutions (int): maximum number of solutions to discover (different solutions
        can be displayed, but only the first one is returned)
    * check_solutions (bool): self-check that the solution is correct (for debugging)
    * print_solutions (bool): print solutions
    * timeout_per_call (int): time limit for sat solver query in milliseconds
    * max_conflicts_per_call (int): max conflicts for sat solver query
    * max_unsolved_depths (int): maximum number of depths without sat/unsat answer

    """

    def run(
        self, high_level_object, coupling_map=None, target=None, qubits=None, **options
    ):
        """Run synthesis for the given `Clifford`."""
        verbosity = options.get("verbosity", 0)
        state_preparation_mode = options.get("state_preparation_mode", False)
        min_depth2q = options.get("min_depth", 0)
        max_depth2q = options.get("max_depth2q", np.inf)
        allow_layout_permutation = options.get("allow_layout_permutation", False)
        allow_final_permutation = options.get("allow_final_permutation", False)
        max_2q_per_layer = options.get("max_2q_per_layer", None)
        max_1q_per_layer = options.get("max_1q_per_layer", None)
        max_num_2q_gates = options.get("max_num_2q_gates", None)
        max_num_1q_gates = options.get("max_num_1q_gates", None)
        max_num_unique_layers = options.get("max_num_2q_gates", None)
        optimize_2q_gates = options.get("optimize_2q_gates", False)
        optimize_1q_gates = options.get("optimize_1q_gates", False)
        restrict_search_space = options.get("restrict_search_space", True)
        max_solutions = options.get("max_solutions", 1)
        check_solutions = options.get("check_solutions", True)
        print_solutions = options.get("print_solutions", False)
        timeout_per_call = options.get("timeout_per_call", None)
        max_conflicts_per_call = options.get("max_conflicts_per_call", None)
        max_unsolved_depths = options.get("max_unsolved_depths", 0)

        if not isinstance(high_level_object, Clifford):
            raise Exception(
                "SatSynthesisCliffordDepth plugin only works with objects of type Clifford."
            )

        if coupling_map is None or qubits is None:
            used_coupling_map = CouplingMap.from_full(high_level_object.num_qubits)
        else:
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        result = synthesize_clifford_depth(
            target_clifford=high_level_object,
            coupling_map=used_coupling_map,
            state_preparation_mode=state_preparation_mode,
            max_depth2q=max_depth2q,
            min_depth2q=min_depth2q,
            allow_layout_permutation=allow_layout_permutation,
            allow_final_permutation=allow_final_permutation,
            max_2q_per_layer=max_2q_per_layer,
            max_1q_per_layer=max_1q_per_layer,
            max_num_2q_gates=max_num_2q_gates,
            max_num_1q_gates=max_num_1q_gates,
            max_num_unique_layers=max_num_unique_layers,
            optimize_2q_gates=optimize_2q_gates,
            optimize_1q_gates=optimize_1q_gates,
            check_solutions=check_solutions,
            print_solutions=print_solutions,
            max_solutions=max_solutions,
            restrict_search_space=restrict_search_space,
            timeout_per_call=timeout_per_call,
            max_conflicts_per_call=max_conflicts_per_call,
            max_unsolved_depths=max_unsolved_depths,
            verbosity=verbosity,
        )

        return result.circuit


class SatSynthesisCliffordCount(HighLevelSynthesisPlugin):
    """Synthesis plugin that uses SAT to find a depth-optimal
    Clifford circuit implementing a given Clifford. The name of
    the plugin is ``clifford.sat_count``.

    In more detail, this plugin is used to synthesize Qiskit objects
    of type `Clifford`. When synthesis succeeds, the plugin
    outputs a quantum circuit consisting of CX, S and H gates.

    The plugin supports the following additional options:

    * verbosity (int): output verbosity (0 = do not show any output)
    * state_preparation_mode (bool): synthesize state rather than the full clifford
    * min_depth2q (int): minimum depth to consider
    * max_depth2q (int): maximum depth to consider
    * allow_layout_permutation (bool): synthesize up to a layout permutation
    * allow_final_permutation (bool): synthesize up to a final permutation
    * max_2q_per_layer (int | None): maximum number of 2q-gates per layer
    * max_1q_per_layer (int | None): maximum number of 1q-gates per layer
    * max_num_2q_gates (int | None): maximum total number of 2a-gates
    * max_num_1q_gates (int | None): maximum total number of 1q-gates
    * max_num_unique_layers (int | None): maximum number of unique layers
    * optimize_2q_gates (bool): after finding the optimal depth, additionally minimize
        the total number of 2-qubit gates for this depth
    * optimize_1q_gates (bool): after minimizing the 2q-depth and the number of 2q-gates,
        additionally minimizes the number of 1q-gates
    * restrict_search_space (bool): use optimizations for restricting search space
    * max_solutions (int): maximum number of solutions to discover (different solutions
        can be displayed, but only the first one is returned)
    * check_solutions (bool): self-check that the solution is correct (for debugging)
    * print_solutions (bool): print solutions
    * timeout_per_call (int): time limit for sat solver query in milliseconds
    * max_conflicts_per_call (int): max conflicts for sat solver query
    * max_unsolved_depths (int): maximum number of depths without sat/unsat answer

    """

    def run(
        self, high_level_object, coupling_map=None, target=None, qubits=None, **options
    ):
        """Run synthesis for the given `Clifford`."""

        verbosity = options.get("verbosity", 0)
        state_preparation_mode = options.get("state_preparation_mode", False)
        min_depth2q = options.get("min_depth", 0)
        max_depth2q = options.get("max_depth2q", np.inf)
        allow_layout_permutation = options.get("allow_layout_permutation", False)
        allow_final_permutation = options.get("allow_final_permutation", False)
        max_2q_per_layer = options.get("max_2q_per_layer", None)
        max_1q_per_layer = options.get("max_1q_per_layer", None)
        max_num_2q_gates = options.get("max_num_2q_gates", None)
        max_num_1q_gates = options.get("max_num_1q_gates", None)
        max_num_unique_layers = options.get("max_num_2q_gates", None)
        optimize_2q_gates = options.get("optimize_2q_gates", False)
        optimize_1q_gates = options.get("optimize_1q_gates", False)
        restrict_search_space = options.get("restrict_search_space", True)
        max_solutions = options.get("max_solutions", 1)
        check_solutions = options.get("check_solutions", True)
        print_solutions = options.get("print_solutions", False)
        timeout_per_call = options.get("timeout_per_call", None)
        max_conflicts_per_call = options.get("max_conflicts_per_call", None)
        max_unsolved_depths = options.get("max_unsolved_depths", 0)

        if not isinstance(high_level_object, Clifford):
            raise Exception(
                "SatSynthesisCliffordCount plugin only works with objects of type Clifford."
            )

        if coupling_map is None or qubits is None:
            used_coupling_map = CouplingMap.from_full(high_level_object.num_qubits)
        else:
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        result = synthesize_clifford_count(
            target_clifford=high_level_object,
            coupling_map=used_coupling_map,
            state_preparation_mode=state_preparation_mode,
            max_depth2q=max_depth2q,
            min_depth2q=min_depth2q,
            allow_layout_permutation=allow_layout_permutation,
            allow_final_permutation=allow_final_permutation,
            max_2q_per_layer=max_2q_per_layer,
            max_1q_per_layer=max_1q_per_layer,
            max_num_2q_gates=max_num_2q_gates,
            max_num_1q_gates=max_num_1q_gates,
            max_num_unique_layers=max_num_unique_layers,
            optimize_2q_gates=optimize_2q_gates,
            optimize_1q_gates=optimize_1q_gates,
            check_solutions=check_solutions,
            print_solutions=print_solutions,
            max_solutions=max_solutions,
            restrict_search_space=restrict_search_space,
            timeout_per_call=timeout_per_call,
            max_conflicts_per_call=max_conflicts_per_call,
            max_unsolved_depths=max_unsolved_depths,
            verbosity=verbosity,
        )

        return result.circuit
