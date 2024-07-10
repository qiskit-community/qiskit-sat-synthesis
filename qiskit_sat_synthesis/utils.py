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

"""Various utility functions."""

import numpy as np

from qiskit.circuit.library.generalized_gates import LinearFunction


def make_downward(coupling_map):
    """Orders each edge (a, b) of the coupling map so that a < b,
    and removes duplicate edges."""
    edges = set()
    for e in coupling_map:
        edges.add(tuple([min(e), max(e)]))
    return list(list(e) for e in edges)


def make_upward(coupling_map):
    """Orders each edge (a, b) of the coupling map so that a > b,
    and removes duplicate edges."""
    edges = set()
    for e in coupling_map:
        edges.add(tuple([max(e), min(e)]))
    return list(list(e) for e in edges)


def perm_matrix(perm):
    """Creates a nxn matrix corresponding to a given permutation pattern."""
    nq = len(perm)
    mat = np.zeros((nq, nq), dtype=bool)
    for i in range(nq):
        mat[i, perm[i]] = True
    return mat


def extend_identity_with_ancillas(
    main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
):
    """
    Sets up initial matrix corresponding over given main qubits and dirty and clean
    ancilla qubits.
    """
    assert main_qubits is not None
    assert dirty_ancilla_qubits is not None
    assert clean_ancilla_qubits is not None

    nq = len(main_qubits) + len(dirty_ancilla_qubits) + len(clean_ancilla_qubits)
    new_mat = np.eye(nq)

    for q in clean_ancilla_qubits:
        new_mat[q, q] = 0

    return new_mat


def extend_mat_with_ancillas(
    mat, main_qubits, dirty_ancilla_qubits, clean_ancilla_qubits
):
    """
    Adjusts mat over main, dirty ancilla and clean ancilla qubits.
    """
    assert main_qubits is not None
    assert dirty_ancilla_qubits is not None
    assert clean_ancilla_qubits is not None
    assert mat.shape == (len(main_qubits), len(main_qubits))

    nq = len(main_qubits) + len(dirty_ancilla_qubits) + len(clean_ancilla_qubits)
    new_mat = np.zeros((nq, nq), dtype=int)
    for i, q in enumerate(main_qubits):
        for j, u in enumerate(main_qubits):
            new_mat[q, u] = mat[i, j]
    for q in dirty_ancilla_qubits:
        new_mat[q, q] = 1
    return new_mat


def _inverse_pattern(pattern):
    """Finds inverse of a permutation pattern."""
    b_map = {pos: idx for idx, pos in enumerate(pattern)}
    return [b_map[pos] for pos in range(len(pattern))]


def _get_ordered_swap(permutation_in):
    """Sorts the input permutation by iterating through the permutation list
    and putting each element to its correct position via a SWAP (if it's not
    at the correct position already). If ``n`` is the length of the input
    permutation, this requires at most ``n`` SWAPs.
    More precisely, if the input permutation is a cycle of length ``m``,
    then this creates a quantum circuit with ``m-1`` SWAPs (and of depth ``m-1``);
    if the input  permutation consists of several disjoint cycles, then each cycle
    is essentially treated independently.
    """
    permutation = list(permutation_in[:])
    swap_list = []
    index_map = _inverse_pattern(permutation_in)
    for i, val in enumerate(permutation):
        if val != i:
            j = index_map[i]
            swap_list.append((i, j))
            permutation[i], permutation[j] = permutation[j], permutation[i]
            index_map[val] = j
            index_map[i] = i
    swap_list.reverse()
    return swap_list
