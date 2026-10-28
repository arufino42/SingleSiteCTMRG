using Revise
using SingleSiteCTMRG
using ITensors

function S(a)
    return 3-2*a
end

function square_ising_local_tensor(T::Float64,J::Float64,h::Float64)
    @assert T != 0.0 && J != 0
    E(a1,a2,a3,a4)=J*(S(a1)*S(a2)+S(a2)*S(a3)+S(a3)*S(a4)+S(a4)*S(a1))-h/2*(S(a1)+S(a2)+S(a3)+S(a4))
    if J>0
        E0=E(1,2,1,2)
    else
        E0=E(1,1,1,1)
    end
    A=Array{Float64,4}(undef,2,2,2,2)
    ops=Dict{String,Any}()
    ops["e"]=Array{Float64,4}(undef,2,2,2,2)
    for a1 in 1:2, a2 in 1:2, a3 in 1:2, a4 in 1:2
        A[a1,a2,a3,a4]=exp( -(E(a1,a2,a3,a4)-E0)/T) 
        ops["e"][a1,a2,a3,a4]=E(a1,a2,a3,a4)*exp( -(E(a1,a2,a3,a4)-E0)/T) 
    end
    return (A,ops)
end

Tc=1/(log(1+sqrt(2))/2)
T=Tc+0.1
J=-1.0
h=0.0
vchi=[10,20]
(A,ops)=square_ising_local_tensor(Tc,J,h)
net=peps(A,vchi[1];ops_in=ops)
out=run_ctmrg(net;vchi)