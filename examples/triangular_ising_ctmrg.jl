using SingleSiteCTMRG
using ITensors
using JLD2

include("../models/triangular.jl")

#Choose initial parameters
# h=parse(Float64,ARGS[1])
# t0=parse(Float64,ARGS[2])
# chi= parse(Int,ARGS[3])
# method= directional,symmetricOrus,symmetricNishino
# Boundary conditions: Open, SymmetryBroken, RandomComplex, Load
# If the "Load" boundary condition is chosen, a 7'th argument can be supplied to use a previously converged environment as the boundary condition.
# Tensor constructions: Hex, SquareSym, SquareSingle

for k=1
    temp=0.
    if length(ARGS)>=1
        temp=parse(Float64,ARGS[1])
    end
    h=0.
    if length(ARGS)>=2
        h=parse(Float64,ARGS[2])
    end
    chi=40
    if length(ARGS)>=3
        chi=parse(Int,ARGS[3])
    end
    method=:symmetricNishino
    if length(ARGS)>=4
        method=Symbol(ARGS[4])
    end
    boundaryCondition=:SymmetryBroken
    if length(ARGS)>=5
        boundaryCondition=Symbol(ARGS[5])
    end
    tensorConstruction=:Hex
    if length(ARGS)>=6
        tensorConstruction=Symbol(ARGS[6])
    end
    InitialisationFile="ChangeSign/h=3_temp=1.34.in"
    if boundaryCondition==:Load
        if length(ARGS)<7
            @warn "No initialisation file given to Load boundary condition. Assuming $InitialisationFile"
        else
            InitialisationFile=ARGS[7]
        end
    end
    nA=0
    @show temp
    @show h
    @show chi
    @show method
    @show boundaryCondition
    @show tensorConstruction
    (curNet,E,OP,M)=triangle_initialiser(temp,h,chi,boundaryCondition,tensorConstruction,InitialisationFile)
    vMag=ssMag(curNet,temp,h,tensorConstruction)
    eTol=1e-9
    fTol=1e-12
    opTol=1e-10
    if nA==0
        corrTol=Inf
    else
        corrTol=1e-6
    end
    vData = []
    Nmin=300
    Nmax=50
    deltaE=2*eTol
    deltaF=2*fTol
    deltaOP=2*opTol
    e1=0
    e2=0
    z1=0
    z2=0
    f1=0
    f2=0
    op1=0
    op2=0
    global convOP=[]
    global convE=[]
    global convF=[]
    global convVals=[]
    global convMag=[]
    global mag=[]
    global Vss=nothing
    # make_finite(curNet)
    make_finite(curNet)
    calc_env(curNet)
    println("z initial = ",partition_function(curNet))
    i=0
    # while (i <= Nmin ||abs(deltaOP)>=opTol||abs(deltaE)>=eTol||abs(deltaF)>=fTol) && i<=Nmax
    while (i <= Nmin || abs(deltaE)>=eTol||abs(deltaOP)>=opTol||abs(deltaF)>=fTol) && i<=Nmax
        i+=1
        # @show curNet.A
        try
            (curNet,vVals)=iterate_ctmrg(curNet,method)
        catch e
            println(e.msg)
            break
        end
        z1=z2
        z2=partition_function(curNet)
        vecMax=make_finite(curNet)
        Vss=corr_function(curNet,vMag[1],vMag[7],1,100)
        s=entanglement_entropy(curNet)
        calc_env(curNet)
        f1=f2
        f2=free_energy(curNet)/3
        deltaF=f2-f1
        e2=average(curNet,E)
        deltaE=e2-e1
        e1=e2
        op2=average(curNet,OP)
        deltaOP=abs(op2)-abs(op1)
        op1=op2
        push!(convOP,op2)
        push!(convE,e2)
        push!(convF,f2)
        mag=[average(curNet,vMag[k]) for k=1:7]
        push!(convMag,mag)
        m2=average(curNet,M)
        # make_finite2(curNet)
        if i%20==0&&i<=200
            try
                vals,vecs =transfer_matrix_eigenvalues(curNet,6)
                push!(convVals,(i,vals))
            catch
                println("Impossible to calculate transfer matrix eigenvalues")
            end
        end
        if i<50||i%20==0
            # @show vecMax
            # @show log(vecMax[2]*vecMax[4]*vecMax[6]*vecMax[8])/(4*3)
            println("i = ",i)
            # @show Vss
            # @show s
            println("h = ", h)
            println("z = ",z2)
            println("deltaZ = ",abs(z2-z1))
            println("f = ",f2)
            println("deltaF = ",deltaF)
            println("e = ",e2)
            println("deltaE = ",deltaE)
            for k=1:7
                println("Mag $k = ",mag[k])
            end
        end
    end
    folder=mkpath("results/Triangular")*"/"
    save_object(folder*"net_h=$(h)_temp=$(temp)_chi=$(chi)_met=$(method)_bc=$(boundaryCondition)_tc=$(tensorConstruction).dat",curNet)
    dataPoint=Dict{String,Any}()
    dataPoint["h"]=h;
    dataPoint["temp"]=temp;
    dataPoint["chi"]=chi;
    dataPoint["deltaE"] =deltaE
    dataPoint["deltaOP"] =deltaOP
    dataPoint["deltaF"] =deltaF
    dataPoint["Niter"]  =i
    dataPoint["convOP"]  =convOP
    dataPoint["convE"]  =convE
    try
        vals,vecs =transfer_matrix_eigenvalues(curNet,15)
        dataPoint["vals"]        =vals
    catch
        println("Transfer matrix eigenvalues not calculated")
    end
    dataPoint["convF"] =convF
    dataPoint["f"]=f2
    dataPoint["convVals"]=convVals
    dataPoint["e"]           =e2
    dataPoint["op"]          =op2
    dataPoint["convMag"]=convMag
    dataPoint["mag"]=mag
    save_object(folder*"data_h=$(h)_temp=$(temp)_chi=$(chi)_met=$(method)_bc=$(boundaryCondition)_tc=$(tensorConstruction).dat",dataPoint)
end
