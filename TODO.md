
### ToDo (no particular order):

- re-examine this to-do

- experiment with other SAT-solvers and SMT-solvers:
  - e.g., PySAT has Cadical
  - CryptoMiniSat has python API
- experiment with higher-level encodings into Z3, possibly let Z3 handle optimization itself

- add functionality to print linear matrices/clifford tableaus after every step in the solution
  (this can be either a debug feature of cnfization, or post-factum analysis of the obtained circuit)

- Explore other high-level schemes when searching for optimal solutions
  - Now when we optimize depth, we use bottom-up (aka UNSAT->SAT) approach
  - And we optimize count given optimal depth, we use top-down (aka SAT->UNSAT) approach
  - There are other schemes based on binary search

- improved search space reduction: add more constraints that may help in various scenarios 
(though it would be the responsibility of the calling application to correctly choose which 
constraints to add). As an example, for depth-minimal clifford synthesis we can require that 
no 2-qubit gate can be moved to a previous layer (i.e. at least one of the qubits is used).

