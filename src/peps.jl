
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
    ops::Dict{String,ITensor}
    configurationDecomposition
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
    return peps(Array{ITensor}(undef,4),Array{ITensor}(undef,4),chi,ITensor(),Array{Index}(undef,4),Array{Index}(undef,4),Array{Index}(undef,4),nothing,Dict{String,ITensor}(),nothing)
end