"""
    bundle_indices(T::ITensor, i1::Index, i2::Index, k::Index) -> ITensor

Replace legs `i1`, `i2` by `k` using a dense, column-major reshape with
`i1` varying fastest. Requires `dim(k) == dim(i1)*dim(i2)`, otherwise
throws `ArgumentError`. Allocates a new tensor. Internal helper; inverse
of [`unbundle_indices`](@ref).
"""
function bundle_indices(T::ITensor,i1::Index, i2::Index, k::Index)
    if dim(i1)*dim(i2)!=dim(k)
        throw(ArgumentError("Dimensions of indices to be bundled don't match"))
    end
    indSet = [i for i in noncommoninds(T,ITensor(i1,i2))]
    nArray= reshape(Array(T,indSet...,i1,i2),[dim(j) for j in indSet]..., dim(k))
    outT = ITensor(nArray,indSet...,k)
    return outT
end

"""
    unbundle_indices(T::ITensor, i1::Index, i2::Index, k::Index) -> ITensor

Split leg `k` into `i1`, `i2` using the ordering of
[`bundle_indices`](@ref). Requires `dim(k) == dim(i1)*dim(i2)`, otherwise
throws `ArgumentError`. Allocates a new dense tensor. Internal helper.
"""
function unbundle_indices(T::ITensor,i1::Index, i2::Index, k::Index)
    if dim(i1)*dim(i2)!=dim(k)
        throw(ArgumentError("Dimensions of indices to be unbundled don't match"))
    end
    indSet = [i for i in noncommoninds(T,ITensor(k))]
    nArray= reshape(Array(T,indSet...,k),[dim(j) for j in indSet]..., dim(i1),dim(i2))
    outT = ITensor(nArray,indSet...,i1,i2)
    return outT
end


"""
    singular_values(curNet, i, nvals=20) -> Vector

Request `nvals` leading singular values of corner `curNet.C[i]` using
KrylovKit with tolerance `1e-20` and Krylov dimension `nvals + 1`.
Choose `i` in `1:4` and a request compatible with the corner dimensions.
Returns values only, discarding convergence information. Not exported.
"""
function singular_values(curNet,i,nvals=20)
    vals,vecs,info= svdsolve(array(curNet.C[i]),nvals,:LR,tol=1e-20,krylovdim=nvals+1)
    return vals
end


"""
    convergence_measure(net::peps) -> Number

Print and return a growth ratio `R1/R2`, formed from the right boundary
`C[2]*E[2]*C[3]` and an added column `E[1]*A*E[3]`. Uses direct tensor
products without complex conjugation. This normalization-dependent
diagnostic does not implement a convergence criterion or use `net.env`.
Not exported.
"""
function convergence_measure(net::peps)
    Psi= net.C[2]*net.E[2]*net.C[3]
    Psi2=Psi*net.E[1]*net.A*net.E[3]
    R1=(replaceinds(Psi,[net.alpha[2],net.i[2],net.alpha[3]],[net.alpha[1],net.i[4],net.alpha[4]])*Psi2)[]
    R2=(Psi*Psi)[]
    @show R1/R2
    return R1/R2;
end