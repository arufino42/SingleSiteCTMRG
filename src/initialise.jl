```
    Function that initializes the CTMRG environment. If no boundaryCondition is given, "Open" boundary conditions are used, where the outside indices are summed over. A tuple of 4 vectors can also be given, which define the boundary condition in each of the 4 directions.
```
function peps(A_in::Array{Float64,4},chi::Int;ops_in=Dict{String,Any}(),boundaryCondition="Open")
    d=size(A_in)[1]
    i=[Index(d) for k in 1:4]
    A=ITensor(A_in,i)
    ops=Dict{String,ITensor}()
    for (name,op) in ops_in
        ops[name]=ITensor(op,i...)
    end

    alpha=[Index(d,tags="alpha$i") for i=1:4]
    beta=[Index(d,tags="beta$i") for i=1:4]

    E=Array{ITensor}(undef,4)
    E[1]=ITensor(0.,alpha[1],alpha[2],inds(A)[1])
    E[2]=ITensor(0.,beta[2],beta[3],inds(A)[2])
    E[3]=ITensor(0.,alpha[3],alpha[4],inds(A)[3])
    E[4]=ITensor(0.,beta[4],beta[1],inds(A)[4])
    C = [ITensor(0.,alpha[i],beta[i]) for i=1:4]

    if boundaryCondition=="Open"
        for i1=1:d,i2=1:d,i3=1:d,i4=1:d
            C[1][alpha[1]=>i2,beta[1]=>i3]+=A[i1,i2,i3,i4]
            C[2][beta[2]=>i3,alpha[2]=>i4]+=A[i1,i2,i3,i4]
            C[3][alpha[3]=>i4,beta[3]=>i1]+=A[i1,i2,i3,i4]
            C[4][beta[4]=>i1,alpha[4]=>i2]+=A[i1,i2,i3,i4]
            E[1][alpha[2]=>i2,inds(A)[1]=>i3,alpha[1]=>i4]+=A[i1,i2,i3,i4]
            E[2][beta[3]=>i3,inds(A)[2]=>i4,beta[2]=>i1]  +=A[i1,i2,i3,i4]
            E[3][alpha[4]=>i4,inds(A)[3]=>i1,alpha[3]=>i2]+=A[i1,i2,i3,i4]
            E[4][beta[1]=>i1,inds(A)[4]=>i2,beta[4]=>i3]  +=A[i1,i2,i3,i4]
        end
    else
        bc = boundaryCondition
        for i1=1:d,i2=1:d,i3=1:d,i4=1:d
            C[1][alpha[1]=>i2,beta[1]=>i3]+=A[i1,i2,i3,i4]*bc[4][i4]*bc[1][i1]
            C[2][beta[2]=>i3,alpha[2]=>i4]+=A[i1,i2,i3,i4]*bc[1][i1]*bc[2][i2]
            C[3][alpha[3]=>i4,beta[3]=>i1]+=A[i1,i2,i3,i4]*bc[2][i2]*bc[3][i3]
            C[4][beta[4]=>i1,alpha[4]=>i2]+=A[i1,i2,i3,i4]*bc[3][i3]*bc[4][i4]
            E[1][alpha[2]=>i2,inds(A)[1]=>i3,alpha[1]=>i4]+=A[i1,i2,i3,i4]*bc[1][i1]
            E[2][beta[3]=>i3,inds(A)[2]=>i4,beta[2]=>i1]  +=A[i1,i2,i3,i4]*bc[2][i2]
            E[3][alpha[4]=>i4,inds(A)[3]=>i1,alpha[3]=>i2]+=A[i1,i2,i3,i4]*bc[3][i3]
            E[4][beta[1]=>i1,inds(A)[4]=>i2,beta[4]=>i3]  +=A[i1,i2,i3,i4]*bc[4][i4]
        end
    end
    return peps(E,C,chi,A,alpha,beta,i,nothing,ops)
end
