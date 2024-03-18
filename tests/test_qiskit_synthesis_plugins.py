import numpy as np

from qiskit.quantum_info import Operator, Clifford
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.generalized_gates import PermutationGate, LinearFunction
from qiskit.transpiler.passes.synthesis.high_level_synthesis import (
    HighLevelSynthesis,
    HLSConfig,
)
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.compiler import transpile

from qiskit_sat_synthesis.qiskit_synthesis_plugins import (
    SatSynthesisPermutationDepth,
    SatSynthesisPermutationCount,
    SatSynthesisLinearFunctionDepth,
    SatSynthesisLinearFunctionCount,
    SatSynthesisCliffordDepth,
    SatSynthesisCliffordCount,
)

# GLOBAL OPTIONS
verbosity = 0
print_solutions = False
check_solutions = True


class TestSynthesizePermutationDepth:
    """Tests for SatSynthesisPermutationDepth"""

    @staticmethod
    def _generate_hls_config_via_class():
        """Generates HLSConfig referring to synthesis plugin via class"""
        hls_config = HLSConfig(
            permutation=[
                (
                    SatSynthesisPermutationDepth(),
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _generate_hls_config_via_name():
        hls_config = HLSConfig(
            permutation=[
                (
                    "sat_depth",
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    def test_run(self):
        """Run plugin directly - without coupling map"""
        perm = PermutationGate([1, 2, 3, 4, 0])
        tqc = SatSynthesisPermutationDepth().run(
            perm,
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc) == Operator(perm)

    def test_run_with_coupling_map(self):
        """Run plugin directly - with coupling map"""
        perm = PermutationGate([1, 2, 3, 4, 0])
        coupling_map = CouplingMap.from_line(8)
        tqc = SatSynthesisPermutationDepth().run(
            perm,
            coupling_map=coupling_map,
            qubits=[7, 6, 4, 5, 3],
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc) == Operator(perm)

    def test_high_level_synthesis(self):
        """Run using HighLevelSynthesis pass - without coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        tqc = hls_pass(qc)
        assert Operator(qc) == Operator(tqc)

    def test_high_level_synthesis_coupling_map(self):
        """Run using HighLevelSynthesis pass - with coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        tqc = hls_pass(qc)
        assert Operator(qc) == Operator(tqc)

    def test_pass_manager(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc) == Operator(tqc)

    def test_pass_manager_with_coupling_map(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(7)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 5, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc) == Operator(tqc)

    def test_transpile(self):
        """Run via transpile - without coupling map"""
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config)
        assert Operator(qc) == Operator(tqc)

    def test_transpile_with_coupling_map(self):
        """Run via transpile - with coupling map"""
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(6)  # need same dimensions
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config, coupling_map=coupling_map)
        assert Operator(qc) == Operator.from_circuit(tqc)


class TestSynthesizePermutationCount:
    """Tests for SatSynthesisPermutationCount"""

    @staticmethod
    def _generate_hls_config_via_class():
        """Generates HLSConfig referring to synthesis plugin via class"""
        hls_config = HLSConfig(
            permutation=[
                (
                    SatSynthesisPermutationCount(),
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _generate_hls_config_via_name():
        hls_config = HLSConfig(
            permutation=[
                (
                    "sat_count",
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    def test_run(self):
        """Run plugin directly - without coupling map"""
        perm = PermutationGate([1, 2, 3, 4, 0])
        tqc = SatSynthesisPermutationCount().run(
            perm,
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc) == Operator(perm)

    def test_run_with_coupling_map(self):
        """Run plugin directly - with coupling map"""
        perm = PermutationGate([1, 2, 3, 4, 0])
        coupling_map = CouplingMap.from_line(8)
        tqc = SatSynthesisPermutationCount().run(
            perm,
            coupling_map=coupling_map,
            qubits=[7, 6, 4, 5, 3],
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc) == Operator(perm)

    def test_high_level_synthesis(self):
        """Run using HighLevelSynthesis pass - without coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        tqc = hls_pass(qc)
        assert Operator(qc) == Operator(tqc)

    def test_high_level_synthesis_coupling_map(self):
        """Run using HighLevelSynthesis pass - with coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        tqc = hls_pass(qc)
        assert Operator(qc) == Operator(tqc)

    def test_pass_manager(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc) == Operator(tqc)

    def test_pass_manager_with_coupling_map(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(7)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 5, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc) == Operator(tqc)

    def test_transpile(self):
        """Run via transpile - without coupling map"""
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config)
        assert Operator(qc) == Operator(tqc)

    def test_transpile_with_coupling_map(self):
        """Run via transpile - with coupling map"""
        qc = QuantumCircuit(6)
        qc.append(PermutationGate([1, 2, 3, 4, 0]), [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(6)  # need same dimensions
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config, coupling_map=coupling_map)
        assert Operator(qc) == Operator.from_circuit(tqc)


class TestSynthesizeLinearFunctionDepth:
    """Tests for SatSynthesisLinearFunctionDepth"""

    @staticmethod
    def _generate_hls_config_via_class():
        """Generates HLSConfig referring to synthesis plugin via class"""
        hls_config = HLSConfig(
            linear_function=[
                (
                    SatSynthesisLinearFunctionDepth(),
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _generate_hls_config_via_name():
        hls_config = HLSConfig(
            linear_function=[
                (
                    "sat_depth",
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _create_linear_function():
        mat = np.array(
            [
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [1, 0, 1, 1, 0],
            ]
        )
        return LinearFunction(mat)

    def test_run(self):
        """Run plugin directly - without coupling map"""
        lf = self._create_linear_function()
        tqc = SatSynthesisLinearFunctionDepth().run(
            lf,
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc) == Operator(lf)

    def test_run_with_coupling_map(self):
        """Run plugin directly - with coupling map"""
        lf = self._create_linear_function()
        coupling_map = CouplingMap.from_line(8)
        tqc = SatSynthesisLinearFunctionDepth().run(
            lf,
            coupling_map=coupling_map,
            qubits=[7, 6, 4, 5, 3],
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc) == Operator(lf)

    def test_high_level_synthesis(self):
        """Run using HighLevelSynthesis pass - without coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        tqc = hls_pass(qc)
        assert Operator(qc) == Operator(tqc)

    def test_high_level_synthesis_coupling_map(self):
        """Run using HighLevelSynthesis pass - with coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        tqc = hls_pass(qc)
        assert Operator(qc) == Operator(tqc)

    def test_pass_manager(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc) == Operator(tqc)

    def test_pass_manager_with_coupling_map(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(7)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc) == Operator(tqc)

    def test_transpile(self):
        """Run via transpile - without coupling map"""
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config)
        assert Operator(qc) == Operator(tqc)

    def test_transpile_with_coupling_map(self):
        """Run via transpile - with coupling map"""
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(6)  # need same dimensions
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config, coupling_map=coupling_map)
        # The line below does not work: I believe it's a problem in `transpile`.
        # Uncomment the line when that problem is fixed.
        # assert Operator(qc) == Operator.from_circuit(tqc)

    def test_solver_limit(self):
        """Run via transpile - without coupling map"""
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])

        hls_config = HLSConfig(
            linear_function=[
                (
                    "sat_depth",
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                        "max_conflicts_per_call": 10,
                    },
                )
            ]
        )
        coupling_map = CouplingMap.from_line(8)

        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc) == Operator(tqc)


class TestSynthesizeLinearFunctionCount:
    """Tests for SatSynthesisLinearFunctionCount"""

    @staticmethod
    def _generate_hls_config_via_class():
        """Generates HLSConfig referring to synthesis plugin via class"""
        hls_config = HLSConfig(
            linear_function=[
                (
                    SatSynthesisLinearFunctionCount(),
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _generate_hls_config_via_name():
        hls_config = HLSConfig(
            linear_function=[
                (
                    "sat_count",
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _create_linear_function():
        mat = np.array(
            [
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 0],
                [0, 0, 0, 1, 1],
                [1, 0, 1, 1, 0],
            ]
        )
        return LinearFunction(mat)

    def test_run(self):
        """Run plugin directly - without coupling map"""
        lf = self._create_linear_function()
        tqc = SatSynthesisLinearFunctionCount().run(
            lf,
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc) == Operator(lf)

    def test_run_with_coupling_map(self):
        """Run plugin directly - with coupling map"""
        lf = self._create_linear_function()
        coupling_map = CouplingMap.from_line(8)
        tqc = SatSynthesisLinearFunctionCount().run(
            lf,
            coupling_map=coupling_map,
            qubits=[7, 6, 4, 5, 3],
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc) == Operator(lf)

    def test_high_level_synthesis(self):
        """Run using HighLevelSynthesis pass - without coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        tqc = hls_pass(qc)
        assert Operator(qc) == Operator(tqc)

    def test_high_level_synthesis_coupling_map(self):
        """Run using HighLevelSynthesis pass - with coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        tqc = hls_pass(qc)
        assert Operator(qc) == Operator(tqc)

    def test_pass_manager(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc) == Operator(tqc)

    def test_pass_manager_with_coupling_map(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(7)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc) == Operator(tqc)

    def test_transpile(self):
        """Run via transpile - without coupling map"""
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config)
        assert Operator(qc) == Operator(tqc)

    def test_transpile_with_coupling_map(self):
        """Run via transpile - with coupling map"""
        qc = QuantumCircuit(6)
        lf = self._create_linear_function()
        qc.append(lf, [4, 2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(6)  # need same dimensions
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config, coupling_map=coupling_map)
        # The line below does not work: I believe it's a problem in `transpile`.
        # Uncomment the line when that problem is fixed.
        # assert Operator(qc) == Operator.from_circuit(tqc)


class TestSynthesizeCliffordDepth:
    """Tests for SatSynthesisCliffordDepth"""

    @staticmethod
    def _generate_hls_config_via_class():
        """Generates HLSConfig referring to synthesis plugin via class"""
        hls_config = HLSConfig(
            clifford=[
                (
                    SatSynthesisCliffordDepth(),
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _generate_hls_config_via_name():
        hls_config = HLSConfig(
            clifford=[
                (
                    "sat_depth",
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _create_clifford():
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.s(0)

        qc.swap(0, 1)
        qc.cx(2, 1)
        qc.cx(1, 2)

        qc.swap(2, 3)

        return Clifford(qc)

    def test_run(self):
        """Run plugin directly - without coupling map"""
        cliff = self._create_clifford()
        tqc = SatSynthesisCliffordDepth().run(
            cliff,
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc).equiv(Operator(cliff))

    def test_run_with_coupling_map(self):
        """Run plugin directly - with coupling map"""
        cliff = self._create_clifford()
        coupling_map = CouplingMap.from_line(8)
        tqc = SatSynthesisCliffordDepth().run(
            cliff,
            coupling_map=coupling_map,
            qubits=[7, 6, 4, 5],
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc).equiv(Operator(cliff))

    def test_high_level_synthesis(self):
        """Run using HighLevelSynthesis pass - without coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        tqc = hls_pass(qc)
        assert Operator(qc).equiv(Operator(tqc))

    def test_high_level_synthesis_coupling_map(self):
        """Run using HighLevelSynthesis pass - with coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        tqc = hls_pass(qc)
        assert Operator(qc).equiv(Operator(tqc))

    def test_pass_manager(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc).equiv(Operator(tqc))

    def test_pass_manager_with_coupling_map(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(7)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc).equiv(Operator(tqc))

    def test_transpile(self):
        """Run via transpile - without coupling map"""
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config)
        assert Operator(qc).equiv(Operator(tqc))

    def test_transpile_with_coupling_map(self):
        """Run via transpile - with coupling map"""
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(6)  # need same dimensions
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config, coupling_map=coupling_map)
        # The line below does not work: I believe it's a problem in `transpile`.
        # Uncomment the line when that problem is fixed.
        # assert Operator(qc).equiv(Operator.from_circuit(tqc))


class TestSynthesizeCliffordCount:
    """Tests for SatSynthesisCliffordCount"""

    @staticmethod
    def _generate_hls_config_via_class():
        """Generates HLSConfig referring to synthesis plugin via class"""
        hls_config = HLSConfig(
            clifford=[
                (
                    SatSynthesisCliffordCount(),
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _generate_hls_config_via_name():
        hls_config = HLSConfig(
            clifford=[
                (
                    "sat_count",
                    {
                        "verbosity": verbosity,
                        "print_solutions": print_solutions,
                        "check_solutions": check_solutions,
                    },
                )
            ]
        )
        return hls_config

    @staticmethod
    def _create_clifford():
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.s(0)

        qc.swap(0, 1)
        qc.cx(2, 1)
        qc.cx(1, 2)

        qc.swap(2, 3)

        return Clifford(qc)

    def test_run(self):
        """Run plugin directly - without coupling map"""
        cliff = self._create_clifford()
        tqc = SatSynthesisCliffordCount().run(
            cliff,
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc).equiv(Operator(cliff))

    def test_run_with_coupling_map(self):
        """Run plugin directly - with coupling map"""
        cliff = self._create_clifford()
        coupling_map = CouplingMap.from_line(8)
        tqc = SatSynthesisCliffordCount().run(
            cliff,
            coupling_map=coupling_map,
            qubits=[7, 6, 4, 5],
            verbosity=verbosity,
            print_solutions=print_solutions,
            check_solutions=check_solutions,
        )
        assert Operator(tqc).equiv(Operator(cliff))

    def test_high_level_synthesis(self):
        """Run using HighLevelSynthesis pass - without coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        tqc = hls_pass(qc)
        assert Operator(qc).equiv(Operator(tqc))

    def test_high_level_synthesis_coupling_map(self):
        """Run using HighLevelSynthesis pass - with coupling map
        In addition, the plugin is referred to by name.
        """
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_name()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        tqc = hls_pass(qc)
        assert Operator(qc).equiv(Operator(tqc))

    def test_pass_manager(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(hls_config=hls_config)
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc).equiv(Operator(tqc))

    def test_pass_manager_with_coupling_map(self):
        """Run via pass manager - without coupling map"""
        qc = QuantumCircuit(7)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(8)
        hls_config = self._generate_hls_config_via_class()
        hls_pass = HighLevelSynthesis(
            hls_config=hls_config, coupling_map=coupling_map, use_qubit_indices=True
        )
        pm = PassManager([hls_pass])
        tqc = pm.run(qc)
        assert Operator(qc).equiv(Operator(tqc))

    def test_transpile(self):
        """Run via transpile - without coupling map"""
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config)
        assert Operator(qc).equiv(Operator(tqc))

    def test_transpile_with_coupling_map(self):
        """Run via transpile - with coupling map"""
        qc = QuantumCircuit(6)
        cliff = self._create_clifford()
        qc.append(cliff, [2, 3, 0, 1])
        coupling_map = CouplingMap.from_line(6)  # need same dimensions
        hls_config = self._generate_hls_config_via_class()
        tqc = transpile(qc, hls_config=hls_config, coupling_map=coupling_map)
        # The line below does not work: I believe it's a problem in `transpile`.
        # Uncomment the line when that problem is fixed.
        # assert Operator(qc).equiv(Operator.from_circuit(tqc))
