
# function transfer_matrix(net::peps,x)
#     if length(x)!=dim(net.alpha[2])*dim(net.alpha[3])*dim(net.i[2])
#         throw(ArgumentError("Vector and matrix dimensions don't match"))
#     end
#     k1=Index(dim(net.alpha[2])*dim(net.alpha[3])*dim(net.i[2]))
#     k2=Index(dim(net.alpha[3])*dim(net.i[2]))
#     Xt= ITensor(x,k1)
#     Xt=unbundle_indices(Xt,net.alpha[2],k2,k1)
#     Xt=unbundle_indices(Xt,net.i[2],net.alpha[3],k2)
#     l1=Index(dim(net.alpha[1])*dim(net.alpha[4])*dim(net.i[4]))
#     l2=Index(dim(net.alpha[4])*dim(net.i[4]))
#     res = ((Xt*net.E[1])*net.A)*net.E[3]
#     res = bundle_indices(res,net.i[4],net.alpha[4],l2)
#     res = bundle_indices(res,net.alpha[1],l2,l1)
#     return vector(res)
# end


"""
    transfer_matrix(net::peps, x) -> Vector

Apply the edge-only map obtained by contracting the upper and lower edges on
their local legs, without `net.A`. Input length is
`dim(alpha[2])*dim(alpha[3])`, flattened with `alpha[2]` fastest.
Output is flattened as `(alpha[1], alpha[4])`, first index fastest.
A wrong input length raises `ArgumentError`. Does not use `net.env`.
Internal helper for [`transfer_matrix_eigenvalues`](@ref).
"""
function transfer_matrix(net::peps,x)
    # @show length(x)
    if length(x)!=dim(net.alpha[2])*dim(net.alpha[3])
        throw(ArgumentError("Vector and matrix dimensions don't match"))
    end
    k1=Index(dim(net.alpha[2])*dim(net.alpha[3]))
    Xt= ITensor(x,k1)
    Xt=unbundle_indices(Xt,net.alpha[2],net.alpha[3],k1)
    l1=Index(dim(net.alpha[1])*dim(net.alpha[4]))
    res = Xt*replaceind(net.E[1],net.i[1],net.i[3])*net.E[3]
    res = bundle_indices(res,net.alpha[1],net.alpha[4],l1)
    # println("Calculating transfer matrix")
    # error("stop")
    return vector(res)
end


"""
    transfer_matrix_adj(net::peps, x) -> Vector

Apply the bulk-containing strip `E[1]*A*E[3]` from left to right indices.
Input is flattened as `(alpha[1], i[4], alpha[4])`, output as
`(alpha[2], i[2], alpha[3])`, first coordinate fastest. A wrong input
length raises `ArgumentError`.

Includes a bulk leg and performs no complex conjugation. Despite its name,
this is not the Hermitian adjoint of the current edge-only
[`transfer_matrix`](@ref). The active eigensolver does not use it.
"""
function transfer_matrix_adj(net::peps,x)
    if length(x)!=dim(net.alpha[1])*dim(net.alpha[4])*dim(net.i[4])
        throw(ArgumentError("Vector and matrix dimensions don't match"))
    end
    k1=Index(dim(net.alpha[1])*dim(net.alpha[4])*dim(net.i[4]))
    k2=Index(dim(net.alpha[4])*dim(net.i[4]))
    Xt= ITensor(x,k1)
    Xt=unbundle_indices(Xt,net.alpha[1],k2,k1)
    Xt=unbundle_indices(Xt,net.i[4],net.alpha[4],k2)
    l1=Index(dim(net.alpha[2])*dim(net.alpha[3])*dim(net.i[2]))
    l2=Index(dim(net.alpha[3])*dim(net.i[2]))
    res = ((Xt*net.E[1])*net.A)*net.E[3]
    res = bundle_indices(res,net.i[2],net.alpha[3],l2)
    res = bundle_indices(res,net.alpha[2],l2,l1)
    return vector(res)
end

"""
    transfer_matrix_eigenvalues(net::peps, nvals=11) -> (values, vectors, info)

Request `nvals` eigenvalues of largest magnitude of the edge-only
[`transfer_matrix`](@ref) with KrylovKit `eigsolve`, tolerance `1e-14`.
The map contracts `E[1]` and `E[3]` without `net.A`. Its dimension is
`dim(alpha[2])*dim(alpha[3])`; input and output dimensions must agree.
Choose a compatible `nvals` and inspect `info` for convergence.

Returns unnormalized eigenvalues, right eigenvectors, and KrylovKit
convergence information. No left vectors are computed. The vectors lack
the bulk-leg layout required by [`form_factors`](@ref). Does not use
`net.env`. Solver exceptions become `Error calculating transfer_matrix`.
"""
function transfer_matrix_eigenvalues(net::peps,nvals=11)
    f1=x->transfer_matrix(net,x)
    f2=x->transfer_matrix_adj(net,x)
    try
        # vals,vecs,info= bieigsolve((f1,f2),dim(net.alpha[2])*dim(net.i[1])*dim(net.alpha[3]),nvals,:LM,tol=1e-14)
        vals,vecs,info=eigsolve(f1,dim(net.alpha[2])*dim(net.alpha[3]),nvals,:LM,tol=1e-14)
        return (vals,vecs,info)
    catch
        error("Error calculating transfer_matrix")
    end
    
end 

"""
    corr_lenght(vals, n1=1, n2=2, n3=3) -> (xi, delta)

Return `xi = -1/log(abs(vals[n2]/vals[n1]))` and
`delta = log(abs(vals[n2]/vals[n3]))` from an ordered transfer spectrum.
`xi` is in transfer-step units. Supply valid indices and nonzero values;
a finite positive length requires `abs(vals[n1]) > abs(vals[n2]) > 0`.
No degeneracy checks are made. The spelling `corr_lenght` is the existing
API name. Not exported.
"""
function corr_lenght(vals,n1=1,n2=2,n3=3)
    return (-1/log(abs(vals[n2]/vals[n1])),log(abs(vals[n2]/vals[n3])))
end

"""
    form_factors(net::peps, o1::ITensor, o2::ITensor, vecs) -> Vector

Contract insertions with paired lists `vecs = (right_vectors, left_vectors)`.
Right vectors are flattened as `(alpha[2], i[2], alpha[3])` and left vectors
as `(alpha[1], i[4], alpha[4])`, first coordinate fastest. The two lists must
have equal length and dominant mode first. Insertions carry `net.A`'s indices.

Returns products `<L1|E1 o1 E3|Rk> * <Lk|E1 o2 E3|R1>` as direct
contractions. Performs no conjugation, normalization, partition-factor
division, or disconnected subtraction. Supply left vectors and normalization
in the convention needed by your calculation. Wrong vector lengths raise
`ArgumentError`.

The current [`transfer_matrix_eigenvalues`](@ref) returns only right vectors
of an edge-only map; its output cannot be used directly here.
"""
function form_factors(net::peps,o1::ITensor,o2::ITensor,vecs)
    VecsRight=[]
    VecsLeft=[]
    for k in eachindex(vecs[1])

        x=vecs[1][k]
        if length(x)!=dim(net.alpha[2])*dim(net.alpha[3])*dim(net.i[2])
            throw(ArgumentError("Vector and matrix dimensions don't match"))
        end
        k1=Index(dim(net.alpha[2])*dim(net.alpha[3])*dim(net.i[2]))
        k2=Index(dim(net.alpha[3])*dim(net.i[2]))
        Xt= ITensor(x,k1)
        Xt=unbundle_indices(Xt,net.alpha[2],k2,k1)
        Xt=unbundle_indices(Xt,net.i[2],net.alpha[3],k2)
        push!(VecsRight,Xt)

        x=vecs[2][k]
        if length(x)!=dim(net.alpha[1])*dim(net.alpha[4])*dim(net.i[4])
            throw(ArgumentError("Vector and matrix dimensions don't match"))
        end
        k1=Index(dim(net.alpha[1])*dim(net.alpha[4])*dim(net.i[4]))
        k2=Index(dim(net.alpha[4])*dim(net.i[4]))
        Xt= ITensor(x,k1)
        Xt=unbundle_indices(Xt,net.alpha[1],k2,k1)
        Xt=unbundle_indices(Xt,net.i[4],net.alpha[4],k2)
        push!(VecsLeft,Xt)
    end
    form_factors=[ ((VecsLeft[1]*net.E[1]*o1*net.E[3]*VecsRight[k])[])*((VecsLeft[k]*net.E[1]*o2*net.E[3]*VecsRight[1])[]) for k in eachindex(vecs[1])]
end