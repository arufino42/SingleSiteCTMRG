function run_ctmrg(curNet::peps;vchi=nothing,fTol=1e-3,Nmin=10,Nmax=100_000,verbose=true,corr_length=true,method=:directionalCorboz)
    df=DataFrame()
    if vchi === nothing
        vchi=[curNet.chi]
    end
    for chi in vchi
        curNet.chi=chi
        fTol=1e-13
        deltaF=2*fTol
        e1=0
        e2=0
        f1=0
        f2=0
        i=0
        deltaE=NaN
        convF=Float64[]
        corr1=0
        corr2=0
        while (abs(deltaF)>fTol ) && i<Nmax  ||i<Nmin 
            i+=1
            (curNet,cornerSvals)=iterate_ctmrg(curNet,method)
            vecMax=make_finite(curNet)
            calc_env(curNet)
            z2=partition_function(curNet)
            f1=f2
            f2=free_energy(curNet)
            push!(convF,f2)
            deltaF=f2-f1
            # deltaE=e2-e1
            # e1=e2
            # vals,vecs =transfer_matrix_eigenvalues(curNet,15)
            # corr=1/log(abs(vals[1]/vals[2]))
            # push!(convCorr,corr)
            if verbose && (i<10||i%100==0)
                println("i = ",i)
                println("f = ", f2)
                println("deltaF = ", deltaF)
                println("z =",z2)
                for (name,op) in curNet.ops
                    av=average(curNet,op)
                    println("$name = ",av)
                end
                if corr_length
                    vals,vecs =transfer_matrix_eigenvalues(curNet,15)
                    corr2=1/log(abs(vals[1]/vals[2]))
                    deltaCorr=corr2-corr1
                    corr1=corr2
                    println("corr = ",corr2) 
                end
                
            end
        end
        dataPoint=Dict{String,Any}()
        dataPoint["chi"]=chi
        dataPoint["Niter"]=i
        dataPoint["f"]=f2
        dataPoint["deltaF"]=deltaF
        dataPoint["convF"]=convF
        for (name,op) in curNet.ops
            av=average(curNet,op)
            dataPoint[name]=av
        end
        if corr_length
            vals,vecs =transfer_matrix_eigenvalues(curNet,15)
            corr2=1/log(abs(vals[1]/vals[2]))
            dataPoint["corr"]=corr2
            dataPoint["vals"]=vals
        end
        push!(df,dataPoint,cols=:union)
    end
    return df
end