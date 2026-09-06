"""
    SingleSiteCTMRG

Single-site corner transfer matrix renormalization group (CTMRG) with ITensors.
Initialize a [`peps`](@ref), grow it with [`iterate_ctmrg`](@ref), normalize
with [`make_finite2`](@ref), and refresh [`calc_env`](@ref) before measuring
observables. Model tensors and convergence checks are supplied by the caller.
See the README for a complete example and the tensor index convention.
"""
module SingleSiteCTMRG

using Printf
using Statistics
using ITensors
using LinearAlgebra
using GenericLinearAlgebra
using DelimitedFiles
using KrylovKit
using Dates
using TensorCast

export peps,make_finite2,calc_env,iterate_ctmrg,partition_function,free_energy,transfer_matrix_eigenvalues,corr_function,average,form_factors,sample_configuration

"""
    peps

Mutable CTMRG state for a square network with one repeated rank-four tensor
`A`. The local indices `i[1:4]` are ordered up, right, down, left, and `chi`
is the maximum retained boundary bond dimension.

Corners `C[1:4]` are upper left, upper right, lower right, lower left.
Edges `E[1:4]` are up, right, down, left. Their index sets are:

| Tensor | Indices |
|:--|:--|
| `C[1]` | `(alpha[1], beta[1])` |
| `C[2]` | `(alpha[2], beta[2])` |
| `C[3]` | `(alpha[3], beta[3])` |
| `C[4]` | `(alpha[4], beta[4])` |
| `E[1]` | `(alpha[1], alpha[2], i[1])` |
| `E[2]` | `(beta[2], beta[3], i[2])` |
| `E[3]` | `(alpha[3], alpha[4], i[3])` |
| `E[4]` | `(beta[4], beta[1], i[4])` |

`env` caches the boundary contraction; refresh it with [`calc_env`](@ref)
after boundary changes. `M` and `entropy` are storage fields initialized to
`nothing` and `0`; CTMRG does not update them. Shared legs must use the same
ITensor `Index` objects, not merely matching tags.
"""
mutable struct peps
    E::Array{ITensor}
    C::Array{ITensor}
    chi::Int
    A::ITensor
    alpha::Array{Index}
    beta::Array{Index}
    i
    env::Union{ITensor,Nothing}
    M::Union{ITensor,Nothing}
    entropy::Number
end


"""
    peps(chi::Int) -> peps

Allocate an uninitialized state with cutoff `chi`. Use a positive cutoff.
The four entries of `C`, `E`, `alpha`, `beta`, and `i` are unassigned,
`A` is empty, and `env` is `nothing`. Assign all tensors and indices before
calling CTMRG or observable routines. This does not build a model or choose
boundary conditions.
"""
function peps(chi::Int)
    return peps(Array{ITensor}(undef,4),Array{ITensor}(undef,4),chi,ITensor(),Array{Index}(undef,4),Array{Index}(undef,4),Array{Index}(undef,4),nothing,nothing,0)
end

# Utilities
include("utils.jl")

# Transfer-Matrix related functions
include("transfer_matrix.jl")

# Heart of the CTMRG algorithm
include("ctmrg_iterate.jl")

# Code for calculating observables
include("observables.jl")

# Code for sampling snapshots
include("snapshots.jl")

end # module SingleSiteCTMRG
