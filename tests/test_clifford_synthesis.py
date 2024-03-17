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

"""Tests for Clifford synthesis."""

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.transpiler import CouplingMap

from qiskit_sat_synthesis.synthesize_clifford import (
    synthesize_clifford_depth,
    synthesize_clifford_count,
)


def test_clifford_synthesis_lnn_depth():
    """
    Simple Clifford synthesis tests for LNN connectivity.
    """

    qc = QuantumCircuit(4)
    qc.x(0)
    qc.s(1)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(1, 3)

    cliff = Clifford(qc)

    coupling_map = list(CouplingMap.from_line(4).get_edges())
    result = synthesize_clifford_depth(
        target_clifford=cliff,
        coupling_map=coupling_map,
        allow_final_permutation=False,
        allow_layout_permutation=False,
        check_solutions=True,
        verbosity=0,
    )
    assert Clifford(result.circuit) == cliff


def test_clifford_synthesis_lnn_count():
    """
    Simple Clifford synthesis tests for LNN connectivity.
    """

    qc = QuantumCircuit(4)
    qc.x(0)
    qc.s(1)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(0, 2)
    qc.cx(1, 3)

    cliff = Clifford(qc)

    coupling_map = list(CouplingMap.from_line(4).get_edges())
    result = synthesize_clifford_count(
        target_clifford=cliff,
        coupling_map=coupling_map,
        allow_final_permutation=False,
        allow_layout_permutation=False,
        check_solutions=True,
        verbosity=0,
    )
    assert Clifford(result.circuit) == cliff
