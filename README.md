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
* `A`: the local tensor.
* `alpha`,`beta` and `i`: ITensor `Index` objects for the indices in the environment.
* `env`: ITensor with the contracted environment.
* `ops`: Dictionary with all the observables one wishes to compute.

# Running the code
Once the local tensor `A` has been defined, CTMRG (with open boundary conditions) can be initialized using `net = peps(A,chi)`. To run CTMRG until convergence use the function `run_ctmrg`. 

# Sampling of system snapshots
This package can also be used to sample system snapshots according to the probability distribution defined by the Gibbs distribution. To do so, a `configurationDecomposition` must be given as a keyword argument in the initialization of `peps`. A `configurationDecomposition` is a vector containing $\{U(k),R(k),D(k),L(k)\}$ such that 

\(
 A_{i_1,i_2,i_3,i_4}= \sum_k U(k)_{i_1} R(k)_{i_2} D(k)_{i_3} L(k)_{i_4},
\)

where $\sum k$ runs through all possible local configurations for a tensor (for instance, if every tensor accounts for a cluster of 4 sites we'd have $k\in \{1,\dots,16\}$, as in the Ising example). Then run `sample_snapshots` in the desired rectangular region.

In order to draw the sampled snapshots, use `draw_spin_configuration`. `lattice_vectors` contains $(\mathbf{a}_1,\mathbf{a}_2)$ while `sites` contains a list of sites in the unit cell.

# Examples
See `examples/square_ising.jl` and `examples/triangular_ising_ctmrg.jl` (WIP) for examples of usage.