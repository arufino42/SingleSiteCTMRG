using Random

"""
    sample(vProb::Array{Float64}, Nsamples=1) -> Vector

Draw configuration indices from a nonempty probability array using the
default RNG. Probabilities must be finite, nonnegative, and sum to one within
`1e-6`; only the sum is checked. Returns a vector even for one draw.
Internal helper.
"""
function sample(vProb::Array{Float64},Nsamples=1)
    N=length(vProb)
    if abs(sum(vProb)-1)>=1e-6
        error("Probability doesn't sum to 1")
    end
    cumProb=Float64[]
    curCum=0
    for k=1:N
        curCum+=vProb[k]
        push!(cumProb,curCum)
    end
    cumProb[end]=1
    vres=[]
    for j=1:Nsamples
        randFloat=rand(UInt64)/typemax(UInt64)
        randTile=findfirst(x->x>randFloat,cumProb)
        push!(vres,randTile)
    end
    return vres
end


"""
    snapshot_first_row(net::peps, proj_U, proj_R, proj_D, proj_L, Lx) -> Vector

Sample the first row of `Lx >= 1` configuration labels from left to right,
starting with the cached environment. Projector lists follow
[`sample_configuration`](@ref). Requires current `net.env`. Internal helper.
"""
function snapshot_first_row(net::peps,proj_U,proj_R,proj_D,proj_L,Lx)
    row_config=[]
    vProb=Float64[]
    norm=(net.env*net.A)[]
    n_confs=length(proj_U)

    for k=1:n_confs
        if false && sum(proj_U[k])==0
            push!(vProb,0)
        else
            # @show k
            prob=(net.env*proj_U[k]*proj_R[k]*proj_D[k]*proj_L[k])[]
            push!(vProb,prob/norm)
        end
    end
    random_tile=sample(vProb)[1]
    push!(row_config,random_tile)
    A_U=net.C[1]*net.E[1]*proj_U[row_config[1]]
    A_U=replaceind(A_U,net.alpha[2],net.alpha[1])
    A_D=net.C[4]*net.E[3]*proj_D[row_config[1]]
    A_D=replaceind(A_D,net.alpha[3],net.alpha[4])
    A_L=net.E[4]*proj_L[row_config[1]]
    proj_R2=Array{ITensor}(undef,n_confs)
    for k=1:n_confs
        proj_R2[k]=replaceind(proj_R[k],net.i[2],net.i[4])
    end
    for x=2:Lx
        vProb=Float64[]
        # @show inds(A_U)
        norm=(A_L*A_U*A_D*net.E[1]*net.E[3]*net.A*proj_R2[row_config[x-1]]*net.C[2]*net.E[2]*net.C[3])[]
        for k=1:n_confs
            if (proj_R2[row_config[x-1]]*proj_L[k])[]==0
                push!(vProb,0)
            else
                prob=(proj_R2[row_config[x-1]]*proj_L[k])[]*(A_L*A_U*A_D*net.E[1]*proj_U[k]*net.E[3]*proj_D[k]*net.C[2]*net.E[2]*proj_R[k]*net.C[3])[]
                push!(vProb,prob/norm)
            end
        end
        random_tile=sample(vProb)[1]
        push!(row_config,random_tile)
        A_U=A_U*net.E[1]*proj_U[random_tile]
        A_U=replaceind(A_U,net.alpha[2],net.alpha[1])
        A_U=A_U/maximum(A_U)
        A_D=A_D*net.E[3]*proj_D[random_tile]
        A_D=replaceind(A_D,net.alpha[3],net.alpha[4])
        A_D=A_D/maximum(A_D)
    end
    return row_config
end


"""
    initialise_Bt(net::peps, prev_row, proj_U, proj_R, proj_D, proj_L) -> ITensor

Contract the upper boundary conditioned on the nonempty sampled `prev_row`,
relabeling remaining boundary indices for the next row. Projector lists
follow [`sample_configuration`](@ref). Returns an unnormalized tensor
without modifying `net`. Internal helper.
"""
function initialise_Bt(net::peps,prev_row,proj_U,proj_R,proj_D,proj_L)
    Lx=length(prev_row)
    Bt=net.C[1]*net.E[4]*proj_L[prev_row[1]]
    for x=1:Lx
        Bt=Bt*net.E[1]*proj_U[prev_row[x]]
        if x!=Lx
            Bt=replaceind(Bt,net.alpha[2],net.alpha[1])
        end
    end
    Bt=Bt*net.C[2]*net.E[2]*proj_R[prev_row[Lx]]
    Bt=replaceinds(Bt,[net.beta[4],net.beta[3]],[net.beta[1],net.beta[2]])
    return Bt
end


"""
    snapshot_next_row(net::peps, prev_row, Bt::ITensor,
                      proj_U, proj_R, proj_D, proj_L) -> Vector

Sample the next row conditioned on `prev_row` and accumulated upper boundary
`Bt`. Build right boundary contractions, then draw labels from left to right.
The width is `length(prev_row)` and must be positive. Projector lists follow
[`sample_configuration`](@ref). The caller updates `Bt` separately.
Internal helper.
"""
function snapshot_next_row(net::peps,prev_row,Bt::ITensor,proj_U,proj_R,proj_D,proj_L)
    vBr=[]
    n_confs=length(proj_U)
    
    proj_D2=Array{ITensor}(undef,n_confs)
    proj_R2=Array{ITensor}(undef,n_confs)
    for k=1:n_confs
        proj_D2[k]=replaceind(proj_D[k],net.i[3],net.i[1])
        proj_R2[k]=replaceind(proj_R[k],net.i[2],net.i[4])
    end
    Br=net.E[2]*net.C[3]
    push!(vBr,Br)
    Lx=length(prev_row)
    for x=1:Lx-1
        Br=Br*net.A*proj_D2[prev_row[Lx-x+1]]*net.E[3]
        Br=replaceinds(Br,[net.i[4],net.alpha[4]],[net.i[2],net.alpha[3]])
        Br=Br/maximum(Br)
        push!(vBr,Br)
    end
    row_config=[]
    norm=(Bt*net.A*net.E[4]*net.C[4]*net.E[3]*vBr[Lx]*proj_D2[prev_row[1]])[]
    vProb=Float64[]
    for k=1:n_confs
        if (proj_D2[prev_row[1]]*proj_U[k])[]==0
            push!(vProb,0)
        else
            prob=(proj_D2[prev_row[1]]*proj_U[k])[]*(Bt*net.E[4]*proj_L[k]*net.C[4]*net.E[3]*proj_D[k]*vBr[Lx]*proj_R[k])[]
            push!(vProb,prob/norm)
        end
    end
    random_tile=sample(vProb)[1]
    push!(row_config,random_tile)
    Bl=net.E[4]*proj_L[random_tile]
    Bd=net.C[4]*net.E[3]*proj_D[random_tile]
    Bd=replaceind(Bd,net.alpha[3],net.alpha[4])
    for x=2:Lx
        vProb=Float64[]
        norm=(Bt*Bl*Bd*net.E[3]*net.A*proj_D2[prev_row[x]]*proj_R2[row_config[x-1]]*vBr[Lx-x+1])[]
        for k=1:n_confs
            if (proj_D2[prev_row[x]]*proj_U[k])[]==0 || (proj_R2[row_config[x-1]]*proj_L[k])[] ==0
                push!(vProb,0)
            else
                prob=(proj_D2[prev_row[x]]*proj_U[k])[]* (proj_R2[row_config[x-1]]*proj_L[k])[] * (Bt*Bl*Bd*net.E[3]*proj_D[k]*vBr[Lx-x+1]*proj_R[k])[]
                push!(vProb,prob/norm)
            end
        end
        random_tile=sample(vProb)[1]
        push!(row_config,random_tile)
        Bd=Bd*net.E[3]*proj_D[random_tile]
        Bd=replaceind(Bd,net.alpha[3],net.alpha[4])
        Bd=Bd/maximum(Bd)
    end
    return row_config
end


"""
    sample_configuration(curNet::peps, Lx, Ly, proj_U, proj_R, proj_D, proj_L)

Sample a rectangular patch from a converged CTMRG environment. Return an
`Lx`-by-`Ly` integer matrix of configuration labels indexed as `[x, y]`,
with positive integer dimensions. Map labels to model degrees of freedom
in caller code. Rows are sampled sequentially, conditioned on earlier choices.

Each of the four equally long lists contains rank-one ITensors on
`curNet.i[1]`, `curNet.i[2]`, `curNet.i[3]`, `curNet.i[4]`, respectively.
For configuration `k`, `rho_k = proj_U[k]*proj_R[k]*proj_D[k]*proj_L[k]`
contains the full local statistical weight. Require
`sum(rho_k for k) == curNet.A`, compatible overlap matching between neighbors,
and real nonnegative conditional probabilities. Include the weight once
across the four factors.

Call [`calc_env`](@ref) first. Uses `Float64` probabilities and the default
RNG; prints row numbers and timing after the first row. The internal sampler
requires probabilities to sum to one within `1e-6`.
See Sec. S1, Eqs. (S1)-(S2), of the supplementary information to
[Rufino et al. (2026)](https://arxiv.org/html/2505.05889v2#S1a).
"""
function sample_configuration(curNet::peps,Lx,Ly,proj_U,proj_R,proj_D,proj_L)
    rand_conf=Array{Int,2}(undef,Lx,Ly)
    first_row=snapshot_first_row(curNet,proj_U,proj_R,proj_D,proj_L,Lx)
    rand_conf[:,1]=first_row
    Bt=initialise_Bt(curNet,first_row,proj_U,proj_R,proj_D,proj_L)
    Bt=Bt/maximum(Bt)
    prev_row=first_row
    for y=2:Ly
        @show y  
        @time next_row=snapshot_next_row(curNet,prev_row,Bt,proj_U,proj_R,proj_D,proj_L)
        rand_conf[:,y]=next_row
        Bt=Bt*curNet.E[4]*proj_L[next_row[1]]*curNet.E[2]*proj_R[next_row[Lx]]
        Bt=replaceinds(Bt,[curNet.beta[4],curNet.beta[3]],[curNet.beta[1],curNet.beta[2]])
        Bt=Bt/maximum(Bt)
        prev_row=next_row
    end
    return rand_conf
end