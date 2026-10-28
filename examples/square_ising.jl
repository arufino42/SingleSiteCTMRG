using Revise
using SingleSiteCTMRG
using ITensors
using DataFrames
using Plots

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
    ops["m"]=Array{Float64,4}(undef,2,2,2,2)
    ops["m_stag"]=Array{Float64,4}(undef,2,2,2,2)
    for a1 in 1:2, a2 in 1:2, a3 in 1:2, a4 in 1:2
        A[a1,a2,a3,a4]=exp( -(E(a1,a2,a3,a4)-E0)/T) 
        ops["e"][a1,a2,a3,a4]=E(a1,a2,a3,a4)/2*exp( -(E(a1,a2,a3,a4)-E0)/T)
        ops["m"][a1,a2,a3,a4]=(S(a1)+S(a2)+S(a3)+S(a4))/4*exp( -(E(a1,a2,a3,a4)-E0)/T) 
        ops["m_stag"][a1,a2,a3,a4]=(S(a1)-S(a2)+S(a3)-S(a4))/4*exp( -(E(a1,a2,a3,a4)-E0)/T) 
    end
    return (A,ops)
end

function exactIsingM(T::Number)
    beta=1/T
    betac=log(1+sqrt(2))/2
    return (beta>betac) ? (1-(sinh(2*beta))^(-4))^(1/8) : 0
end

Tc=1/(log(1+sqrt(2))/2)
vT= Tc-0.5:0.05:Tc+0.5

J=-1.0
h=0.0
vchi=[10,20]



if J<0
    bc=repeat([[1,0]],4)
else
    bc=[
        [1,0],
        [0,1],
        [1,0],
        [0,1]
    ]
end


df=DataFrame()
for T in vT
    # Initialising the local tensor, observables and environments
    (A,ops)=square_ising_local_tensor(T,J,h)
    net=peps(A,vchi[1];ops_in=ops,boundaryCondition=bc)

    # Running CTMRG
    out=run_ctmrg(net;vchi)
    out.T .= T
    append!(df,out,cols=:union)
end

sort!(df,:T)
gdf=groupby(df,[:chi])


# Plotting the magnetization
p1=scatter(;xaxis="\$T\$",yaxis="\$m\$",legendtitle="\$\\chi\$")
plot!(p1,exactIsingM,Tc-0.5,Tc+0.5;label="")
for chi in vchi
    scatter!(p1, gdf[(chi=chi,)].T,gdf[(chi=chi,)].m,label="$chi")
end
display(p1)