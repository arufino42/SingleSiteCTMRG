
"""
    extend_left(net::peps, method) -> (net, U)

Absorb a column into the left boundary using `:directionalOrus` or
`:directionalNishino`. Mutates the left corners, edge, and indices, retaining
at most `net.chi` states. Returns the same state and the full left
singular-vector matrix `U`, not singular values. Does not normalize tensors
or refresh `net.env`. Internal helper.
"""
function extend_left(net::peps,method)
    nC1 = net.E[1]*net.C[1]
    nC4 = net.E[3]*net.C[4]
    TC1=zeros(typeof(net.C[1][1,1]),dim(net.beta[1])*dim(net.i[1]),dim(net.alpha[2]))
    TC4=zeros(typeof(net.C[1][1,1]),dim(net.beta[4])*dim(net.i[3]),dim(net.alpha[3]))
    for beta1=1:dim(net.beta[1]),i1=1:dim(net.i[1]),alpha2=1:dim(net.alpha[2])
        TC1[i1+(beta1-1)*dim(net.i[1]),alpha2]=nC1[net.beta[1]=>beta1,net.i[1]=>i1,net.alpha[2]=>alpha2]
    end
    for beta4=1:dim(net.beta[4]),i3=1:dim(net.i[3]),alpha3=1:dim(net.alpha[3])
        TC4[i3+(beta4-1)*dim(net.i[3]),alpha3]=nC4[net.beta[4]=>beta4,net.i[3]=>i3,net.alpha[3]=>alpha3]
    end
    # @show inds(net.C[2])
    if method==:directionalOrus
        M=TC1*TC1'+TC4*TC4'
    elseif method==:directionalNishino
        M=(TC1*Array(net.C[2],net.alpha[2],net.beta[2])*Array(net.C[3],net.beta[3],net.alpha[3])*TC4')'
    end
    F=svd(M)
    cutInd= findfirst(x->abs(x/last(F.S))<=2e-64,F.S)
    if cutInd==nothing
        cutInd=net.chi
    end
    Z= F.U[:,1:min(net.chi,dim(net.i[1])*dim(net.beta[1]),cutInd)]
    betaL1=Index(size(Z,2);tags="beta1");
    tZ = ITensor(typeof(net.C[1][1,1]),net.i[1],net.beta[1],betaL1)
    for i=1:dim(net.i[1]),beta=1:dim(net.beta[1]),betaL=1:dim(betaL1)
        tZ[i,beta,betaL] = Z[i+(beta-1)*dim(net.i[1]) ,betaL]
    end
    tZl = ITensor(typeof(net.C[1][1,1]),net.i[1],net.beta[1],betaL1)
    for i=1:dim(net.i[1]),beta=1:dim(net.beta[1]),betaL=1:dim(betaL1)
        tZl[i,beta,betaL] = conj(Z[i+(beta-1)*dim(net.i[1]) ,betaL])
    end
    betaL4=Index(size(Z,2);tags="beta4");
    tZ2 = ITensor(typeof(net.C[1][1,1]),net.i[3],net.beta[4],betaL4)
    for i=1:dim(net.i[3]),beta=1:dim(net.beta[4]),betaL=1:dim(betaL4)
        tZ2[i,beta,betaL] = Z[i+(beta-1)*dim(net.i[3]) ,betaL]
    end
    tZl2 = ITensor(typeof(net.C[1][1,1]),net.i[3],net.beta[4],betaL4)
    for i=1:dim(net.i[3]),beta=1:dim(net.beta[4]),betaL=1:dim(betaL4)
        tZl2[i,beta,betaL] = conj(Z[i+(beta-1)*dim(net.i[3]) ,betaL])
    end
    net2=net
    net2.C[1]=net.E[1]*net.C[1]*tZl
    net2.C[4]=net.E[3]*net.C[4]*tZ2
    net2.E[4]=net.A*net.E[4]*tZ*tZl2
    net2.C[1]=replaceind(net.C[1],net.alpha[2],net.alpha[1])
    net2.C[4]=replaceind(net.C[4],net.alpha[3],net.alpha[4])
    net2.E[4]=replaceind(net.E[4],net.i[2],net.i[4])
    net2.beta[1] = betaL1
    net2.beta[4] = betaL4
    return net2,F.U
end


"""
    extend_symmetric(net::peps, method) -> (s1, s2, s3, s4)

Grow all corners and edges together using `:symmetricOrus` or
`:symmetricNishino`. Mutates `net`, retaining at most `net.chi` states,
and returns the four untruncated SVD spectra. Assumes equal local dimensions
and a common boundary dimension for reshaping. Does not normalize tensors
or refresh `net.env`. Internal helper.
"""
function extend_symmetric(net::peps,method)
    d=dim(net.i[1])
    b=dim(net.alpha[1])
    nc1=net.C[1]*net.E[1]*net.E[4]*net.A
    nc2=net.C[2]*net.E[1]*net.E[2]*net.A
    nc3=net.C[3]*net.E[3]*net.E[2]*net.A
    nc4=net.C[4]*net.E[3]*net.E[4]*net.A
    ne1=net.E[1]*net.A
    ne2=net.E[2]*net.A
    ne3=net.E[3]*net.A
    ne4=net.E[4]*net.A
    mc1=Array(nc1,net.beta[4],net.i[3],net.alpha[2],net.i[2])
    mc2=Array(nc2,net.alpha[1],net.i[4],net.beta[3],net.i[3])
    mc3=Array(nc3,net.beta[2],net.i[1],net.alpha[4],net.i[4])
    mc4=Array(nc4,net.alpha[3],net.i[2],net.beta[1],net.i[1])
    # In order to make the reshaping of tensors more efficient, we use TensorCast.jl
    @cast TC1[(x1,j1),(x2,j2)]:=mc1[x1,j1,x2,j2]
    @cast TC2[(x1,j1),(x2,j2)]:=mc2[x1,j1,x2,j2]
    @cast TC3[(x1,j1),(x2,j2)]:=mc3[x1,j1,x2,j2]
    @cast TC4[(x1,j1),(x2,j2)]:=mc4[x1,j1,x2,j2]
    M=Array{Any,1}(UndefInitializer(), 4)
    F=Array{Any,1}(UndefInitializer(), 4)
    if method==:symmetricOrus
        M[1]=TC1'*TC1+TC2*TC2'
        M[2]=TC2'*TC2+TC3*TC3'
        M[3]=TC3'*TC3+TC4*TC4'
        M[4]=TC4'*TC4+TC1*TC1'
    elseif method==:symmetricNishino
        M[1]=(TC2*TC3*TC4*TC1)'
        M[2]=(TC3*TC4*TC1*TC2)'
        M[3]=(TC4*TC1*TC2*TC3)'
        M[4]=(TC1*TC2*TC3*TC4)'
    end
    Threads.@threads for k=1:4
        F[k]=svd(M[k],alg = LinearAlgebra.QRIteration())
    end
    # @show F[1].S
    ci1=findfirst(x->abs(x/first(F[1].S))<=2^(-64),F[1].S)
    ci2=findfirst(x->abs(x/first(F[2].S))<=2^(-64),F[2].S)
    ci3=findfirst(x->abs(x/first(F[3].S))<=2^(-64),F[3].S)
    ci4=findfirst(x->abs(x/first(F[4].S))<=2^(-64),F[4].S)
    # @show (ci1,ci2,ci3,ci4)
    if ci1==nothing ||ci2==nothing ||ci3==nothing ||ci4==nothing
        cutInd=net.chi
    else
        cutInd=max(ci1,ci2,ci3,ci4)
    end
    # @show cutInd
    U1=F[1].U[:,1:min(net.chi,b*d,cutInd)]
    U2=F[2].U[:,1:min(net.chi,b*d,cutInd)]
    U3=F[3].U[:,1:min(net.chi,b*d,cutInd)]
    U4=F[4].U[:,1:min(net.chi,b*d,cutInd)]
    vVals=(F[1].S,F[2].S,F[3].S,F[4].S)
    bL=size(U1,2)
    gamma1=Index(bL,tags="gamma1")
    gamma2=Index(bL,tags="gamma2")
    gamma3=Index(bL,tags="gamma3")
    gamma4=Index(bL,tags="gamma4")
    @cast rU1[x,j,g]:= U1[(x,j),g] x in 1:b, j in 1:d
    @cast rU2[x,j,g]:= U2[(x,j),g] x in 1:b, j in 1:d
    @cast rU3[x,j,g]:= U3[(x,j),g] x in 1:b, j in 1:d
    @cast rU4[x,j,g]:= U4[(x,j),g] x in 1:b, j in 1:d
    tU1=ITensor(rU1,net.alpha[2],net.i[2],gamma1)
    tU1d=ITensor(conj.(rU1),net.alpha[1],net.i[4],gamma1')
    tU2=ITensor(rU2,net.beta[3],net.i[3],gamma2)
    tU2d=ITensor(conj.(rU2),net.beta[2],net.i[1],gamma2')
    tU3=ITensor(rU3,net.alpha[4],net.i[4],gamma3)
    tU3d=ITensor(conj.(rU3),net.alpha[3],net.i[2],gamma3')
    tU4=ITensor(rU4,net.beta[1],net.i[1],gamma4)
    tU4d=ITensor(conj.(rU4),net.beta[4],net.i[3],gamma4')
    net.alpha[1]=Index(bL,tags="alpha1")
    net.alpha[2]=Index(bL,tags="alpha2")
    net.alpha[3]=Index(bL,tags="alpha3")
    net.alpha[4]=Index(bL,tags="alpha4")
    net.beta[1]=Index(bL,tags="beta1")
    net.beta[2]=Index(bL,tags="beta2")
    net.beta[3]=Index(bL,tags="beta3")
    net.beta[4]=Index(bL,tags="beta4")
    net.C[1]=replaceinds(nc1*tU1*tU4d,[gamma4',gamma1],[net.beta[1],net.alpha[1]])
    net.C[2]=replaceinds(nc2*tU2*tU1d,[gamma1',gamma2],[net.alpha[2],net.beta[2]])
    net.C[3]=replaceinds(nc3*tU3*tU2d,[gamma2',gamma3],[net.beta[3],net.alpha[3]])
    net.C[4]=replaceinds(nc4*tU4*tU3d,[gamma3',gamma4],[net.alpha[4],net.beta[4]])
    net.E[1]=replaceinds(ne1*tU1d*tU1,[gamma1',gamma1,net.i[3]],[net.alpha[1],net.alpha[2],net.i[1]])
    net.E[2]=replaceinds(ne2*tU2d*tU2,[gamma2',gamma2,net.i[4]],[net.beta[2],net.beta[3],net.i[2]])
    net.E[3]=replaceinds(ne3*tU3d*tU3,[gamma3',gamma3,net.i[1]],[net.alpha[3],net.alpha[4],net.i[3]])
    net.E[4]=replaceinds(ne4*tU4d*tU4,[gamma4',gamma4,net.i[2]],[net.beta[4],net.beta[1],net.i[4]])
    return vVals
end


"""
    extend_left_corboz(net::peps, method) -> (net, singular_values)

Grow the left boundary with projectors from the SVD of two enlarged half
environments. Mutates `net` and returns its full singular-value vector.
`method` is accepted for the directional interface but is not inspected.
Assumes compatible equal dimensions for reshaping; does not normalize tensors
or refresh `net.env`. Internal helper for `:directionalCorboz`.
"""
function extend_left_corboz(net::peps,method)
    d=dim(net.i[1])
    b=dim(net.beta[1])
    nc1=net.C[1]*net.E[1]*net.E[4]*net.A
    nc2=net.C[2]*net.E[1]*net.E[2]*net.A
    nc3=net.C[3]*net.E[3]*net.E[2]*net.A
    nc4=net.C[4]*net.E[3]*net.E[4]*net.A
    # ne1=net.E[1]*net.A
    # ne2=net.E[2]*net.A
    # ne3=net.E[3]*net.A
    # ne4=net.E[4]*net.A
    mc1=Array(nc1,net.beta[4],net.i[3],net.alpha[2],net.i[2])
    mc2=Array(nc2,net.alpha[1],net.i[4],net.beta[3],net.i[3])
    mc3=Array(nc3,net.beta[2],net.i[1],net.alpha[4],net.i[4])
    mc4=Array(nc4,net.alpha[3],net.i[2],net.beta[1],net.i[1])
    # In order to make the reshaping of tensors more efficient, we use TensorCast.jl
    @cast TC1[(x1,j1),(x2,j2)]:=mc1[x1,j1,x2,j2]
    @cast TC2[(x1,j1),(x2,j2)]:=mc2[x1,j1,x2,j2]
    @cast TC3[(x1,j1),(x2,j2)]:=mc3[x1,j1,x2,j2]
    @cast TC4[(x1,j1),(x2,j2)]:=mc4[x1,j1,x2,j2]
    R1=(TC1*TC2)'
    R2=(TC3*TC4)'
    # @show sum(R1*R2)
    # @show norm(R1*R2)
    F=svd(R1*R2,alg = LinearAlgebra.DivideAndConquer())
    # @show F.S
    # @show F
    cutInd=findfirst(x->abs(x/first(F.S))<=16*eps(typeof(F.S[1])),F.S)
    # @show F.S
    # @show b*d
    # @show cutInd
    if cutInd==nothing
        cutInd=b*d
    end
    P1=(R2*F.V*Diagonal(F.S.^(-1/2)))[:,1:min(net.chi,b*d,cutInd-1)]
    P2=(Diagonal(F.S.^(-1/2))*(F.U')*R1)[1:min(net.chi,b*d,cutInd-1),:]
    vVals=(F.S)

    bL=size(P1,2)
    gamma1=Index(bL,tags="gamma1")
    gamma2=Index(bL,tags="gamma2")
    @cast rP1[x,j,g]:= P1[(x,j),g] x in 1:b, j in 1:d
    @cast rP2[g,x,j]:= P2[g,(x,j)] x in 1:b, j in 1:d
    tP1=ITensor(rP1,net.beta[1],net.i[1],gamma1)
    tP1d=ITensor(rP1,net.beta[4],net.i[3],gamma2)
    tP2=ITensor(rP2,gamma2,net.beta[4],net.i[3])
    tP2d=ITensor(rP2,gamma1,net.beta[1],net.i[1])
    net2=net
    net2.beta[1]=Index(bL,tags="beta1")
    net2.beta[4]=Index(bL,tags="beta4")
    net2.C[1]=replaceinds(net.C[1]*net.E[1]*tP1,[gamma1,net.alpha[2]],[net2.beta[1],net.alpha[1]])
    net2.C[4]=replaceinds(net.C[4]*net.E[3]*tP2,[gamma2,net.alpha[3]],[net2.beta[4],net.alpha[4]])
    net2.E[4]=replaceinds(net.E[4]*net.A*tP1d*tP2d,[gamma1,gamma2,net.i[2]],[net2.beta[1],net2.beta[4],net.i[4]])
    return (net,vVals)
end


"""
    rotate_left(net1::peps) -> peps

Return a state with boundary and local-index labels rotated counterclockwise
by a quarter turn: new directions `1:4` are old directions `(2, 3, 4, 1)`.
Tensor objects are shared with `net1`; this is not a deep copy. The bulk
tensor and cached fields are carried over. Internal helper.
"""
function rotate_left(net1::peps)
    net2=peps(net1.chi)
    net2.C[1]=net1.C[2]
    net2.C[2]=net1.C[3]
    net2.C[3]=net1.C[4]
    net2.C[4]=net1.C[1]
    net2.E[1]=net1.E[2]
    net2.E[2]=net1.E[3]
    net2.E[3]=net1.E[4]
    net2.E[4]=net1.E[1]
    net2.alpha[1]=net1.beta[2]
    net2.alpha[2]=net1.beta[3]
    net2.alpha[3]=net1.beta[4]
    net2.alpha[4]=net1.beta[1]
    net2.beta[1]=net1.alpha[2]
    net2.beta[2]=net1.alpha[3]
    net2.beta[3]=net1.alpha[4]
    net2.beta[4]=net1.alpha[1]
    net2.i[1]=net1.i[2]
    net2.i[2]=net1.i[3]
    net2.i[3]=net1.i[4]
    net2.i[4]=net1.i[1]
    net2.A=net1.A
    net2.env=net1.env
    return net2
end


"""
    vertical_reflection(net1::peps) -> peps

Return a state reflected about the vertical axis, exchanging left and right
edges and the corresponding corners and index labels. Tensor objects are
shared with `net1`; the bulk tensor and cached fields are carried over.
This is a relabeling, not a deep copy. Internal helper.
"""
function vertical_reflection(net1::peps)
    net2=peps(net1.chi)
    net2.C[1]=net1.C[2]
    net2.C[2]=net1.C[1]
    net2.C[3]=net1.C[4]
    net2.C[4]=net1.C[3]
    net2.E[1]=net1.E[1]
    net2.E[2]=net1.E[4]
    net2.E[3]=net1.E[3]
    net2.E[4]=net1.E[2]
    net2.alpha[1]=net1.alpha[2]
    net2.alpha[2]=net1.alpha[1]
    net2.alpha[3]=net1.alpha[4]
    net2.alpha[4]=net1.alpha[3]
    net2.beta[1]=net1.beta[2]
    net2.beta[2]=net1.beta[1]
    net2.beta[3]=net1.beta[4]
    net2.beta[4]=net1.beta[3]
    net2.i[1]=net1.i[1]
    net2.i[2]=net1.i[4]
    net2.i[3]=net1.i[3]
    net2.i[4]=net1.i[2]
    net2.A=net1.A
    net2.env=net1.env
    return net2
end


"""
    iterate_ctmrg(net::peps, method) -> (net, diagnostics)

Perform one CTMRG growth step with maximum boundary dimension `net.chi`.
Supported methods and their diagnostics are:

- `:symmetricOrus`, `:symmetricNishino`: grow all directions together;
  return a tuple of four full singular-value vectors.
- `:directionalOrus`, `:directionalNishino`: sweep right, down, left, up;
  return a vector of four full left singular-vector matrices.
- `:directionalCorboz`: sweep right, down, left, up;
  return a vector of four full singular-value vectors.

Growth mutates the input. Always keep the returned state, because directional
sweeps also create rotated state containers. Initialize all tensors first,
with equal local leg dimensions and compatible boundary dimensions.
An unsupported method raises an error.

This performs one step without a convergence test or normalization. Call
[`make_finite`](@ref) after each step and [`calc_env`](@ref) before
measurements; the cached environment is not refreshed automatically.
"""
function iterate_ctmrg(net::peps,method)
    if method==:symmetricOrus || method==:symmetricNishino
         vals=extend_symmetric(net,method)
         return (net,vals)
    elseif method==:directionalOrus||method==:directionalNishino||method==:directionalCorboz
        net,val1=extend_right(net,method)
        net,val2=extend_down(net,method)
        if method ==:directionalCorboz
            net,val3=extend_left_corboz(net,method)
        else
            net,val3=extend_left(net,method)
        end
        net,val4=extend_up(net,method)
        return (net,[val1,val2,val3,val4])
    else
        error("Unrecognized method")
    end
end

"""
    extend_up(net::peps, method) -> (net, diagnostics)

Grow the upper boundary through rotations and a left update. Accepts the
directional methods of [`iterate_ctmrg`](@ref), with the same per-direction
diagnostics. Keep the returned state; normalization and environment refresh
are separate. Internal helper.
"""
function extend_up(net::peps,method)
    net2=rotate_left(net)
    if method==:directionalCorboz
        (net3,values)=extend_left_corboz(net2,method)
    else
        (net3,values)=extend_left(net2,method)
    end
    net3=rotate_left(net3)
    net3=rotate_left(net3)
    net3=rotate_left(net3)
    return net3,values
end

"""
    extend_right(net::peps, method) -> (net, diagnostics)

Grow the right boundary through rotations and a left update. Accepts the
directional methods of [`iterate_ctmrg`](@ref), with the same per-direction
diagnostics. Keep the returned state; normalization and environment refresh
are separate. Internal helper.
"""
function extend_right(net::peps,method)
    net2=rotate_left(net)
    net2=rotate_left(net2)
    if method==:directionalCorboz
        (net3,values)=extend_left_corboz(net2,method)
    else
        (net3,values)=extend_left(net2,method)
    end
    net3=rotate_left(net3)
    net3=rotate_left(net3)
    return net3,values
end

"""
    extend_down(net::peps, method) -> (net, diagnostics)

Grow the lower boundary through rotations and a left update. Accepts the
directional methods of [`iterate_ctmrg`](@ref), with the same per-direction
diagnostics. Keep the returned state; normalization and environment refresh
are separate. Internal helper.
"""
function extend_down(net::peps,method)
    net2=rotate_left(net)
    net2=rotate_left(net2)
    net2=rotate_left(net2)
    if method==:directionalCorboz
        (net3,values)=extend_left_corboz(net2,method)
    else
        (net3,values)=extend_left(net2,method)
    end
    net3=rotate_left(net3)
    return net3,values
end


"""
    calc_env(net::peps) -> ITensor

Contract all four corners and edges, assign the result to `net.env`, and
return it. Its open indices are `net.i[1:4]`, ready to contract with `net.A`
or an observable insertion. Refresh after initialization, growth, or
normalization, before [`average`](@ref), [`partition_function`](@ref),
[`free_energy`](@ref), [`corr_function`](@ref), or
[`sample_configuration`](@ref).
"""
function calc_env(net::peps)
    net.env = ((net.C[1]*net.E[1])*(net.C[2]*net.E[2]))*((net.C[3]*net.E[3])*(net.C[4]*net.E[4]))
end

"""
    make_finite(net::peps) -> scales

Divide each corner and edge in place by its largest absolute entry. Return
the eight divisors in order `[C1, E1, C2, E2, C3, E3, C4, E4]`.
Tensors must have finite entries and nonzero maxima; zero tensors are not
guarded against. Does not change `net.A` or refresh `net.env`.
Call [`calc_env`](@ref) afterwards when measurements are needed.
"""
function make_finite(net::peps)
    # calc_env(net)
    maxVal=[]
    for i=1:4
        push!(maxVal,max(maximum(map(x->abs(x),array(net.C[i])) ),0))
        push!(maxVal,max(maximum(map(x->abs(x),array(net.E[i])) ),0))
        net.C[i]=net.C[i]/max(maximum(map(x->abs(x),array(net.C[i])) ),0)
        net.E[i]=net.E[i]/max(maximum(map(x->abs(x),array(net.E[i])) ),0)
    end
    return (maxVal)
end