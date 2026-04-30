
function average(net::peps,T::ITensor)
    return (net.env*T)[]/partition_function(net)
end


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


