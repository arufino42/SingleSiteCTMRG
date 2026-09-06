using SingleSiteCTMRG
using ITensors
using JLD2
using QuadGK
include("../models/square.jl")
# Metods: symmetric, directional
# Boundary conditions: Open,RandomComplex,RandomReal,SymmetryBroken
# Tensor constructions: Plaquette, SingleSite
# error("test")
Tc=1/(log(1+sqrt(2))/2)
temp=2.3
if length(ARGS)>=1
    temp=parse(Float64,ARGS[1])
end
h=0.
if length(ARGS)>=2
    h=parse(Float64,ARGS[2])
end
chi=20
if length(ARGS)>=3
    chi=parse(Int,ARGS[3])
end
method=:directionalCorboz
if length(ARGS)>=4
    method=Symbol(ARGS[4])
end
boundaryCondition=:SymmetryBroken
if length(ARGS)>=5
    boundaryCondition=Symbol(ARGS[5])
end
tensorConstruction=:SingleSite
if length(ARGS)>=6
    tensorConstruction=Symbol(ARGS[6])
end
@show temp
@show h
@show chi
@show method
@show boundaryCondition
@show tensorConstruction
eTol=1e-9
mTol=1e-9
fTol=1e-9
Nmax=100000
Nmin=1
(curNet,E,M)=initialiseCTMRG_square(chi,temp,h,boundaryCondition,tensorConstruction)
@show curNet.A
@show M
make_finite(curNet)
convM=[]
convE=[]
let deltaF=2*fTol,f1=f2=0,deltaE=2*eTol,e1=0,e2=0,m1=0,m2=0,deltaM=2*mTol,i=0
    while (abs(deltaE)>=eTol||abs(deltaF)>=fTol||abs(deltaM)>=mTol||i<Nmin||abs(imag(m2))>=1e-5)&&i<Nmax
        i+=1
        global curNet
        (curNet,vVals)=iterate_ctmrg(curNet,method)
        vecMax=make_finite(curNet)
        calc_env(curNet)
        f1=f2
        f2=free_energy(curNet)
        deltaF=f2-f1
        if tensorConstruction==:SingleSite
            e2=ferro_energy(curNet,M)
        elseif tensorConstruction==:Plaquette
            e2=average(curNet,E)
        end
        deltaE=abs(e2-e1)
        e1=e2
        m2=average(curNet,M)
        deltaM=abs(m2-m1)
        m1=m2
        push!(convE,e2)
        push!(convM,m2)
        if i%200==0
            @show i
            println("f = ",f2)
            println("deltaF = ",deltaF)
            println("e = ",e2)
            println("deltaE = ",deltaE)
            println("m = ",m2)
            println("deltaM = ",deltaM)
        end
    end
    if i==Nmax
        println("Failure to converge")
    end
    folder=mkpath("examples/results/SquareIsing")*"/"
    save_object(folder*"net_h=$(h)_temp=$(temp)_chi=$(chi)_met=$(method)_bc=$(boundaryCondition)_tc=$(tensorConstruction).dat",curNet)
    println("e = ",e2)
    println("m = ",m2)
    println("deltaE = ",deltaE)
    println("deltaM = ",deltaM)
    dataPoint=Dict{String,Any}()
    dataPoint["h"]=h;
    dataPoint["temp"]=temp;
    dataPoint["chi"]=chi;
    dataPoint["deltaE"] =deltaE
    dataPoint["deltaM"] =deltaM
    dataPoint["e"]      =e2
    dataPoint["m"]      =m2
    dataPoint["convE"]      =convE
    dataPoint["convM"]      =convM
    save_object(folder*"data_h=$(h)_temp=$(temp)_chi=$(chi)_met=$(method)_bc=$(boundaryCondition)_tc=$(tensorConstruction).dat",dataPoint)
end