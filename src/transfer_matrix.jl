
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

function corr_lenght(vals,n1=1,n2=2,n3=3)
    return (-1/log(abs(vals[n2]/vals[n1])),log(abs(vals[n2]/vals[n3])))
end

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