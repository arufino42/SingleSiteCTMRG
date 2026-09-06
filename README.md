# SingleSiteCTMRG.jl

SingleSiteCTMRG contracts an infinite square tensor network with one repeated
local tensor using the corner transfer matrix renormalization group (CTMRG).
It uses ITensors.jl to represent four corner tensors and four edge tensors,
with a maximum environment bond dimension `chi`.

The library provides CTMRG growth steps, local averages, connected horizontal
correlations, transfer spectra, and sampling of finite configuration patches.
You supply the model tensor, initial boundary tensors, observable insertions,
and convergence loop.

## Getting started

Use Julia 1.12 with the dependency versions specified in
[Project.toml](Project.toml). From a checkout:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=.
```

The current checkout imports `Random` in `snapshots.jl` without declaring it
in `Project.toml`, so `using SingleSiteCTMRG` currently fails the package
dependency check. To run the examples from the repository root without changing
the package, load its module directly in a fresh Julia session:

```julia
include("src/SingleSiteCTMRG.jl")
using .SingleSiteCTMRG
using ITensors
```

Docstrings are available in Julia's help mode, for example `?peps` and
`?iterate_ctmrg`. Unexported helpers require qualification, such as
`?SingleSiteCTMRG.corr_lenght`.

## Tensor layout

The local tensor `net.A` has four distinct indices `net.i[1:4]`, ordered
**up, right, down, left**. The corner and edge numbering is:

```text
C[1] -- E[1] -- C[2]
 |        |        |
E[4] ---- A ---- E[2]
 |        |        |
C[4] -- E[3] -- C[3]
```

| Tensor | Index set |
|:--|:--|
| `A` | `i[1], i[2], i[3], i[4]` |
| `C[1]` | `alpha[1], beta[1]` |
| `C[2]` | `alpha[2], beta[2]` |
| `C[3]` | `alpha[3], beta[3]` |
| `C[4]` | `alpha[4], beta[4]` |
| `E[1]` | `alpha[1], alpha[2], i[1]` |
| `E[2]` | `beta[2], beta[3], i[2]` |
| `E[3]` | `alpha[3], alpha[4], i[3]` |
| `E[4]` | `beta[4], beta[1], i[4]` |

Connected tensors must share the same `Index` objects; matching dimensions
and tags alone do not connect them. The growth routines assume equal local
leg dimensions and compatible boundary dimensions. A uniform initial boundary
dimension of one is sufficient for the example below; growth increases it up
to the cutoff.

`peps(chi)` allocates an **uninitialized** container. All entries of `C`,
`E`, `alpha`, `beta`, and `i`, as well as `A`, must be assigned before
use. `env` is a cached contraction populated by `calc_env`; `M` and
`entropy` are storage fields that the iteration does not update.

## Complete example: square-lattice Ising model

This example uses spins `(+1, -1)`, coupling `J = 1`, and `k_B = 1`.
The bond weight is `W[s,t] = exp(K * spin[s] * spin[t])`, where
`K = J / temperature`. Its positive square root `Q` satisfies
`Q * transpose(Q) == W`. Summing the product of four `Q` factors over the
central spin gives the local tensor; weighting that sum by the spin gives
the magnetization insertion.

Run this after loading the module above:

```julia
function ising_environment(; K=0.2, chi=8)
    net = peps(chi)
    net.i = [Index(2, "i$n") for n in 1:4]
    net.alpha = [Index(1, "alpha$n") for n in 1:4]
    net.beta = [Index(1, "beta$n") for n in 1:4]
    i, a, b = net.i, net.alpha, net.beta

    u, v = sqrt(2cosh(K)), sqrt(2sinh(K))
    Q = [(u + v)/2 (u - v)/2; (u - v)/2 (u + v)/2]
    spin = (1.0, -1.0)
    weights = zeros(2, 2, 2, 2)
    magnetization = zeros(2, 2, 2, 2)
    for up in 1:2, right in 1:2, down in 1:2, left in 1:2
        for s in 1:2
            w = Q[s, up] * Q[s, right] * Q[s, down] * Q[s, left]
            weights[up, right, down, left] += w
            magnetization[up, right, down, left] += spin[s] * w
        end
    end
    net.A = ITensor(weights, i...)
    spin_tensor = ITensor(magnetization, i...)

    for n in 1:4
        net.C[n] = ITensor(ones(1, 1), a[n], b[n])
    end
    net.E[1] = ITensor(ones(1, 1, 2), a[1], a[2], i[1])
    net.E[2] = ITensor(ones(1, 1, 2), b[2], b[3], i[2])
    net.E[3] = ITensor(ones(1, 1, 2), a[3], a[4], i[3])
    net.E[4] = ITensor(ones(1, 1, 2), b[4], b[1], i[4])
    calc_env(net)
    return net, spin_tensor, Q
end

function converge_environment(net; method=:symmetricOrus,
                              maxiter=100, tolerance=1e-10)
    previous = free_energy(net)
    for step in 1:maxiter
        net, diagnostics = iterate_ctmrg(net, method)
        make_finite2(net)
        calc_env(net)
        current = free_energy(net)
        if step >= 10 && abs(current - previous) < tolerance
            return net, step
        end
        previous = current
    end
    error("CTMRG did not meet the chosen tolerance within $maxiter steps")
end

net, spin_tensor, Q = ising_environment()
net, steps = converge_environment(net)

log_partition_per_site = free_energy(net)
temperature = 1 / 0.2
physical_free_energy = -temperature * log_partition_per_site
magnetization = average(net, spin_tensor)
connected_correlations = corr_function(net, spin_tensor, spin_tensor, 1, 10)
@show steps log_partition_per_site physical_free_energy magnetization
```

This stopping rule monitors one scalar. For your model, also check observables,
initialization dependence, and convergence with increasing `chi`. The example
uses a spin-symmetric boundary; selecting an ordered state can require a biased
initial boundary.

Always retain the state returned by `iterate_ctmrg`. It mutates tensors, and
directional sweeps also return rotated containers. Growth and normalization
leave the cached environment stale: call `calc_env(net)` before measurements.

## Update methods and measurements

| Method symbol | Update | Second return value |
|:--|:--|:--|
| `:symmetricOrus` | All four directions together | Tuple of four SVD spectra |
| `:symmetricNishino` | All four directions together | Tuple of four SVD spectra |
| `:directionalOrus` | Right, down, left, up | Four full left singular-vector matrices |
| `:directionalNishino` | Right, down, left, up | Four full left singular-vector matrices |
| `:directionalCorboz` | Right, down, left, up | Four SVD spectra |

These diagnostics are untruncated and are not interchangeable between methods.
Each call performs one growth step; there is no built-in convergence loop.
`make_finite2` rescales each corner and edge by its maximum absolute entry and
returns those eight divisors. All boundary tensors must be nonzero.

| Function | Meaning |
|:--|:--|
| `calc_env(net)` | Rebuild and return `net.env` |
| `partition_function(net)` | Local scalar `(net.env * net.A)[]` |
| `average(net, O)` | Weighted insertion divided by the local partition contraction |
| `free_energy(net)` | Log partition ratio per repeated tensor |
| `corr_function(net, O1, O2, rmin, rmax)` | Connected horizontal correlations over the inclusive range |
| `transfer_matrix_eigenvalues(net, nvals=11)` | Edge-only transfer eigenvalues, right vectors, and convergence information |
| `form_factors(net, O1, O2, vecs)` | Insertion products using caller-supplied paired left/right modes |
| `sample_configuration(net, Lx, Ly, U, R, D, L)` | Matrix of sampled local configuration labels |

The observable tensor `O` has the same indices as `A` and includes the local
statistical weights. Passing `A` itself to `average` gives one.

`partition_function` depends on the current boundary normalization; it is not
the infinite system's full partition function. `free_energy` computes
`log(abs(z1*z2/(z3*z4)))`, canceling the corner and edge rescalings. It includes
neither the thermodynamic minus sign nor temperature. Multiply by `-T` and
convert from tensor units to physical site units as appropriate. If you rescale
the bulk tensor, restore its contribution separately.

`corr_function` returns connected correlations at positive integer distances
in tensor spacings, with `r = 1` denoting adjacent tensors. Its intermediate
strip contractions are not rescaled, which limits accessible distances.

### Transfer-spectrum conventions

The active eigensolver contracts the upper and lower edges on their local legs;
it does **not** insert `A`. Returned eigenvalues are not normalized. Choose
`nvals` compatible with `dim(net.alpha[2])*dim(net.alpha[3])` and inspect
the returned KrylovKit `info`.

The unexported `SingleSiteCTMRG.corr_lenght(values)` returns
`(-1/log(abs(values[2]/values[1])), log(abs(values[2]/values[3])))`.
The first quantity is a length in units of this transfer map; interpreting it
as a physical correlation length requires the appropriate model mapping.

`form_factors` expects `(right_vectors, left_vectors)` for a bulk-containing
strip, with three indices per vector. The current eigensolver returns only
right vectors for a two-index edge map, so its output cannot be passed directly
to `form_factors`. See the docstrings for flattening and normalization details.

## Sampling configurations

Supply four equally long lists of rank-one ITensors `U, R, D, L`, on the up,
right, down, and left local indices. For every label `k`, their product must
represent the full weighted local configuration; summing these products must
reproduce `A`. Include each configuration's weight once across the four
factors. Neighboring factors must give the intended overlap weights, and
conditional probabilities must be real and nonnegative.

For the Ising example, each label is one spin and the factors are the rows of
`Q`. After convergence:

```julia
using Random
Random.seed!(42)
projectors = [
    [ITensor(Q[s, :], net.i[direction]) for s in 1:2]
    for direction in 1:4
]
U, R, D, L = projectors
calc_env(net)
labels = sample_configuration(net, 4, 3, U, R, D, L)
spins = map(k -> k == 1 ? 1 : -1, labels)
```

The output has shape `(Lx, Ly)` and indexing `[x, y]`; it contains
configuration labels, not an automatic model-to-spin conversion. Sampling
uses the CTM boundary around the patch, `Float64` probabilities, and the
default random-number generator. It prints row progress and timing. The
internal sampler checks normalization within `1e-6`; inconsistent factors
or an insufficiently converged boundary can fail that check.

## Research use case and citation

A use case of SingleSiteCTMRG is the constrained kagome Ising antiferromagnet
studied by Afonso Rufino, Samuel Nyckees, Jeanne Colbois, and Frédéric Mila in
[“Topological Devil’s Staircase in a Constrained Kagome Ising Antiferromagnet,”
*Physical Review Letters* **136**, 086701 (2026)](https://doi.org/10.1103/dgyr-5y75).

The public [preprint with supplementary information](https://arxiv.org/html/2505.05889v2)
provides the relevant methodological details:

- [End Matter EM2](https://arxiv.org/html/2505.05889v2#S2):
  the kagome-star tensor construction and directional CTMRG environment.
- [Supplement S1](https://arxiv.org/html/2505.05889v2#S1a):
  weighted configuration factors and sequential conditional sampling.
- [Supplement S2](https://arxiv.org/html/2505.05889v2#S2a):
  real-space correlations obtained from sampled configurations.
- [Supplement S3](https://arxiv.org/html/2505.05889v2#S3a):
  bond-dimension dependence and the need to check increasing `chi`.

The paper illustrates how CTMRG observables and snapshots reveal a sequence
of topological plateaus. Its model-specific kagome tensors and analysis must
be supplied separately; the example above demonstrates the library on a
simpler model.

When discussing this research use case, cite:

```bibtex
@article{Rufino2026Topological,
  author  = {Rufino, Afonso and Nyckees, Samuel and Colbois, Jeanne and Mila, Frédéric},
  title   = {Topological Devil's Staircase in a Constrained Kagome Ising Antiferromagnet},
  journal = {Physical Review Letters},
  volume  = {136},
  pages   = {086701},
  year    = {2026},
  doi     = {10.1103/dgyr-5y75}
}
```
