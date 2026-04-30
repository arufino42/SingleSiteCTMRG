
function transfer_matrix(net::peps,x)
    if length(x)!=dim(net.alpha[2])*dim(net.alpha[3])*dim(net.i[2])
        throw(ArgumentError("Vector and matrix dimensions don't match"))
    end
    k1=Index(dim(net.alpha[2])*dim(net.alpha[3])*dim(net.i[2]))
    k2=Index(dim(net.alpha[3])*dim(net.i[2]))
    Xt= ITensor(x,k1)
    Xt=unbundle_indices(Xt,net.alpha[2],k2,k1)
    Xt=unbundle_indices(Xt,net.i[2],net.alpha[3],k2)
    l1=Index(dim(net.alpha[1])*dim(net.alpha[4])*dim(net.i[4]))
    l2=Index(dim(net.alpha[4])*dim(net.i[4]))
    res = ((Xt*net.E[1])*net.A)*net.E[3]
    res = bundle_indices(res,net.i[4],net.alpha[4],l2)
    res = bundle_indices(res,net.alpha[1],l2,l1)
    return vector(res)
end

function transfer_matrix_eigenvalues(net::peps,nvals=11)
    vals,vecs,info= bieigsolve(x->transfer_matrix(net,x),dim(net.alpha[2])*dim(net.i[1])*dim(net.alpha[3]),nvals,:LM,tol=1e-20)
    return (vals,vecs,info)
end 

function corr_lenght(vals,n1=1,n2=2,n3=3)
    return (-1/log(abs(vals[n2]/vals[n1])),log(abs(vals[n2]/vals[n3])))
end