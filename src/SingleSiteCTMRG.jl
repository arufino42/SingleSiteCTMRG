"""
    SingleSiteCTMRG

Single-site corner transfer matrix renormalization group (CTMRG) with ITensors.
Initialize a [`peps`](@ref), grow it with [`iterate_ctmrg`](@ref), normalize
with [`make_finite`](@ref), and refresh [`calc_env`](@ref) before measuring
observables. Model tensors and convergence checks are supplied by the caller.
See the README for a complete example and the tensor index convention.
"""
module SingleSiteCTMRG

using Printf
using ITensors
using LinearAlgebra
using GenericLinearAlgebra
using DelimitedFiles
using KrylovKit
using Dates
using TensorCast
using DataFrames
using Luxor
using JLD2

export peps,make_finite,calc_env,iterate_ctmrg,partition_function,free_energy,transfer_matrix_eigenvalues,corr_function,average,form_factors,sample_snapshots,entanglement_entropy,initialise_open_bc,run_ctmrg,draw_tile, draw_spin_configuration

# Structure definition
include("peps.jl")

# Utilities
include("utils.jl")

# Initialisation
include("initialise.jl")

# Transfer-Matrix related functions
include("transfer_matrix.jl")

# Heart of the CTMRG algorithm
include("ctmrg_iterate.jl")

# Calculating observables
include("observables.jl")

# Sampling snapshots
include("snapshots.jl")

# Drawing snapshots
include("draw.jl")

# Running CTMRG loop
include("run.jl")

end # module SingleSiteCTMRG
