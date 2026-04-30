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

end # module SingleSiteCTMRG
