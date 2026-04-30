function bundle_indices(T::ITensor,i1::Index, i2::Index, k::Index)
    if dim(i1)*dim(i2)!=dim(k)
        throw(ArgumentError("Dimensions of indices to be bundled don't match"))
    end
    indSet = [i for i in noncommoninds(T,ITensor(i1,i2))]
    nArray= reshape(Array(T,indSet...,i1,i2),[dim(j) for j in indSet]..., dim(k))
    outT = ITensor(nArray,indSet...,k)
    return outT
end

function unbundle_indices(T::ITensor,i1::Index, i2::Index, k::Index)
    if dim(i1)*dim(i2)!=dim(k)
        throw(ArgumentError("Dimensions of indices to be unbundled don't match"))
    end
    indSet = [i for i in noncommoninds(T,ITensor(k))]
    nArray= reshape(Array(T,indSet...,k),[dim(j) for j in indSet]..., dim(i1),dim(i2))
    outT = ITensor(nArray,indSet...,i1,i2)
    return outT
end