# SingleSiteCTMRG.jl

SingleSiteCTMRG is a julia implementation the corner transfer matrix renormalization group (CTMRG) method for the approximate contraction of an infinite translationally invariant square tensor network.
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

## Usage examples: 
See `examples/` for examples of implementations of CTMRG on the square and triangular lattices, as well as for usage of the snapshot sampling.

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
`make_finite` rescales each corner and edge by its maximum absolute entry and
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
