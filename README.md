# SingleSiteCTMRG.jl

SingleSiteCTMRG is a julia implementation the corner transfer matrix renormalization group ([CTMRG](https://doi.org/10.1143/JPSJ.65.891)) method for the approximate contraction of an infinite translationally-invariant square tensor network.
It uses ITensors.jl to represent four corner tensors and four edge tensors, with a maximum environment bond dimension `chi`.

# Installing the package
In order to install the package in julia, run

``using Pkg; Pkg.develop(url="https://github.com/arufino42/SingleSiteCTMRG")``.

# Internal representation
The CTMRG environment is saved in the `peps` variable which contains
* `E`: vector with the four edge tensors.
* `C`: vector with the four corner tensors.
* `chi`: the cut-off dimension
* `A`: the local tensor
* `alpha`,`beta` and `i`: ITensor `Index` objects for the indices in the environment.
* `env`: ITensor with the contracted environment
* `ops`: Dictionary with all the observables one wishes to compute.

# Running the code
Once the local tensor `A` has been defined, CTMRG (with open boundary conditions) can be initialized using `net = peps(A,chi)`. To run CTMRG until convergence use the function `run_ctmrg`. 

# Examples
See `examples/square_ising.jl` and `examples/triangular_ising.jl` for examples of usage.