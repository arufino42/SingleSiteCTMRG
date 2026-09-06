
"""
    average(net::peps, T::ITensor) -> Number

Return `(net.env*T)[] / partition_function(net)`. `T` is a local rank-four
observable insertion with the same indices as `net.A`, including the model's
statistical weights, rather than a bare physical operator matrix.
Requires a current environment and nonzero normalization.
"""
function average(net::peps,T::ITensor)
    return (net.env*T)[]/partition_function(net)
end


"""
    corr_function(net::peps, Op1::ITensor, Op2::ITensor, rmin::Int, rmax::Int)

Return the connected horizontal correlation
`<Op1(0) Op2(r)> - <Op1><Op2>` for each integer in `rmin:rmax`.
Use `1 <= rmin <= rmax`; distance is in local tensor spacings and nearest
neighbors have `r = 1`. Both insertions use the indices and statistical-weight
convention of `net.A`.

Requires current `net.env` for one-point averages. The strip contraction
uses upper and lower edges and does not rescale intermediates, so long
distances can overflow or underflow. Returns a vector with element type
taken from `Op1`.
"""
function corr_function(net::peps,Op1::ITensor,Op2::ITensor,rmin::Int, rmax::Int)
    av1=average(net,Op1)
    av2=average(net,Op2)
    Tm = net.C[1]*net.E[1]*net.E[4]*Op1*net.C[4]*net.E[3]
    Tz = net.C[1]*net.E[1]*net.E[4]*net.A*net.C[4]*net.E[3]
    Vss = zeros(typeof(Op1[1]),0)
    for r=1:rmax
        Tm=replaceinds(Tm,[net.alpha[2],net.i[2],net.alpha[3]],[net.alpha[1],net.i[4],net.alpha[4]])
        Tz=replaceinds(Tz,[net.alpha[2],net.i[2],net.alpha[3]],[net.alpha[1],net.i[4],net.alpha[4]])
        z=(Tz*net.E[1]*net.C[2]*net.A*net.E[2]*net.E[3]*net.C[3])[]
        ss =  (Tm*net.E[1]*net.C[2]*Op2*net.E[2]*net.E[3]*net.C[3])[]/z
        if r>=rmin
            push!(Vss,ss - av1*av2)
        end
        Tm=Tm*net.E[1]*net.A*net.E[3]
        Tz=Tz*net.E[1]*net.A*net.E[3]
    end
    return Vss
end


"""
    entanglement_entropy(net::peps) -> Number

Compute `-tr(rho*log(rho))` from the matrix product of corners in order
`C2*C1*C4*C3`, normalized to unit trace. Uses the matrix logarithm and assumes
nonzero trace and a suitable spectrum; no positivity or regularization checks
are applied. Does not use `net.env` or assign `net.entropy`. Not exported.
"""
function entanglement_entropy(net::peps)
    C2=Array(net.C[2],net.beta[2],net.alpha[2])
    C1=Array(net.C[1],net.alpha[1],net.beta[1])
    C4=Array(net.C[4],net.beta[4],net.alpha[4])
    C3=Array(net.C[3],net.alpha[3],net.beta[3])
    rho=C2*C1*C4*C3
    rho=rho/tr(rho)
    s=-tr(rho*log(rho))
    return s
end



"""
    partition_function(net::peps) -> Number

Return `(net.env*net.A)[]` using a current environment from [`calc_env`](@ref).
This local contraction depends on the normalization of the CTM boundary and
is not the full partition function of the infinite lattice. It is the
denominator for [`average`](@ref).
"""
function partition_function(net::peps)
    return (net.env*net.A)[]
end

"""
    free_energy(net::peps) -> Real

Return `log(abs(z1*z2/(z3*z4)))`, the CTM estimate of the log partition weight
per repeated tensor. `z1` includes the bulk tensor and all edges, `z2`
contracts corners only, and `z3`, `z4` include the horizontal and vertical
edge pairs, respectively. Boundary rescaling factors cancel in this ratio.
Requires current `net.env` and nonzero contractions.

No minus sign, temperature, or number-of-spins factor is included. For
Boltzmann weights at temperature `T` with `k_B = 1`, the physical free
energy per tensor is `-T * free_energy(net)`. Use the model's tensor-to-lattice
mapping for per-spin units, restoring any removed bulk weight factors.
"""
function free_energy(net::peps)
    z1=partition_function(net)
    z2=(net.C[1]*replaceind(net.C[2],net.alpha[2],net.alpha[1])*replaceind(net.C[3],net.beta[3],net.beta[2])*replaceinds(net.C[4],[net.alpha[4],net.beta[4]],[net.alpha[3],net.beta[1]]))[]
    z3=(net.C[1]*net.E[1]*net.C[2]*replaceind(net.C[3],net.beta[3],net.beta[2])*replaceind(net.E[3],net.i[3],net.i[1])*replaceind(net.C[4],net.beta[4],net.beta[1]))[]
    z4=(net.C[1]*net.E[4]*net.C[4]*replaceind(net.C[2],net.alpha[2],net.alpha[1])*replaceind(net.E[2],net.i[2],net.i[4])*replaceind(net.C[3],net.alpha[3],net.alpha[4]))[]
    return log(abs(z1*z2/(z3*z4)))
end
