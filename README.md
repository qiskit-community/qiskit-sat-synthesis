# Qiskit SAT Synthesis

This repository contains a collection of SAT-based synthesis methods for various ``Qiskit`` objects,
including objects of type``Clifford``, ``LinearFunction`` and ``PermutationGate``.

## Approach

A synthesis problem such as _"Does there exist a quantum circuit consisting at most ``k`` CNOT gates 
that implements a given ``n x n`` linear function"_ is translated into conjunctive normal form (CNF) 
and is solved using ``Z3`` SMT-solver. The problem is repeatedly solved with increasing values of ``k`` 
until a solution is found.

## Features

* Synthesis of Cliffords, Linear Functions and Permutations.
* Supports arbitrary coupling maps.
* Allows to optimize the number of 2-qubit gates or the 2-qubit depth.
* Allows synthesis up to _layout permutation_, up to _final permutation_ or both.
* Allows to easily integrate other synthesis methods.

## Installation

The package is not yet available through pypi, but can be installed by cloning this repository:

```
git clone https://github.com/qiskit-community/qiskit-sat-synthesis
```

and then installing locally:

```
pip install ./qiskit-sat-synthesis
```

## Basic Usage

Once installed, the ``HighLevelSynthesis`` transpiler pass in Qiskit is able to detect high-level synthesis methods
via an entry point. The following example illustrates this basic usage:

```
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import LinearFunction
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig

# A 5x5 binary invertible matrix corresponding to a long-range CNOT-gate.
mat = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ]

# A quantum circuit which contains a linear function corresponding to our matrix.
qc = QuantumCircuit(5)
qc.append(LinearFunction(mat), [0, 1, 2, 3, 4])

# The coupling map
coupling_map = CouplingMap.from_line(5)

# The high-level synthesis config to synthesize high-level objects in the circuit. Notably,it specifies 
# the "sat_depth" method to synthesize linear functions. The method accepts additional parameters:
# the output verbosity and the option to minimize the number of 2-qubit gates once the minimum depth is 
# found.
config = HLSConfig(linear_function=[("sat_depth", {"verbosity": 1, "optimize_2q_gates": True})])

# Running high-level synthesis transpiler pass and printing the transpiled circuit.
qct = HighLevelSynthesis(hls_config=config, coupling_map=coupling_map, use_qubit_indices=True)(qc)
print(qct)
```

## More information

Plesse check out the python notebook `notebooks/using-qiskit-sat-synthesis.ipunb` and various examples 
in the `examples` directory.

## Limitations

Due to the nature of the approach, only works for reasonably small problems. 
