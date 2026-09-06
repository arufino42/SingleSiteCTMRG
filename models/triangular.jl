
function S(a)
    return 3-2*a
end


function print_square_tile(s)
    println("
    $(s[1]==1 ? "+" : "-" ) ------ $(s[2]==1 ? "+" : "-" )
    |      /  |
    |    /    |
    |  /      |
    $(s[4]==1 ? "+" : "-" ) ------ $(s[3]==1 ? "+" : "-" )")
end

function hex_energy(s0,s,h)
    E=0
    E+= -h*S(s0)
    for i=1:6
        E+= S(s0)*S(s[i])
        E+= -h*S(s[i])/3
        E+= 0.5*S(s[i])*S(s[i%6+1])
    end
    return E
end

function net_load()
    data=readdlm("TensorHoneycomb.dat",Int)
    i1=Index(4)
    i2=Index(4)
    i3=Index(4)
    i4=Index(4)
    matA=reshape(data,4,4,4,4)
    A=ITensor(matA,i1,i2,i3,i4)
    M=E=OP=A
    (A,E,OP,M)
end


function calcDiff(v)
    return [abs(v[i+1]-v[i]) for i=1:length(v)-1]
end

function hex_op(s0,s)
    return  S(s0) + exp(2*pi*im/3)*(S(s[1])+S(s[3])+S(s[5]))/3+exp(4*pi*im/3) *(S(s[2])+S(s[4])+S(s[6]))/3
end

function hex_m(s0,s)
    return  S(s0) +(S(s[1])+S(s[2])+S(s[3])+S(s[4])+S(s[5])+S(s[6]))/3
end

function square_sym_mag(s)
    return  1.0/6*(S(s[1])+S(s[3]))+1.0/3*(S(s[4])+S(s[2]))
end

function square_sym_order_parameter(s)
    return S(s[4])/3+S(s[2])/3*exp(4*pi*im/3) +(S(s[3])+S(s[1]))/6*exp(2*im*pi/3)
end

function square_single_mag(s)
    return  1/3*(S(s[1])+S(s[2])+S(s[4]))
end

function square_single_order_parameter(s)
    return S(s[4])/3+S(s[2])/3*exp(4*pi*im/3) +S(s[1])/3*exp(2*im*pi/3)
end

function bundle(s1,s2)
    return 2*(s1-1)+s2
end


function hex_tile(temp,h)
    i=[Index(4),Index(4),Index(4),Index(4)]
    A=ITensor(i...)
    OP=ITensor(Complex{Float64},i...)
    M=ITensor(Complex{Float64},i...)
    E=ITensor(i...)
    if h<=6
        e0=-3-h
    else
        e0=9-3*h
    end
    i=0
    for s0=1:2,s1=1:2,s2=1:2,s3=1:2,s4=1:2,s5=1:2,s6=1:2
        if temp!=0
            A[bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)
            E[bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)/3
            OP[bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=hex_op(s0,[s1,s2,s3,s4,s5,s6])*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)/3
            M[bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=hex_m(s0,[s1,s2,s3,s4,s5,s6])*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)/3
        elseif temp==0
            if (hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)-e0)<=1e-7
                i=i+1
                A[bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=1
                E[bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/3
                OP[bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=hex_op(s0,[s1,s2,s3,s4,s5,s6])/3
                M[bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=hex_m(s0,[s1,s2,s3,s4,s5,s6])/3
            end
        end
    end
    if temp!=0
        lambda=exp(-e0/temp)
    else
        lambda=1
    end
    # lambda=maximum(A)
    nNonZero=sum(array(A))
    @show nNonZero
    A=A/lambda
    E=E/lambda
    OP=OP/lambda
    M=M/lambda
    return (A,E,OP,M)
    # @show A[bundle(1,1),bundle(1,1),bundle(1,2),bundle(1,1)]
end

function square_sym_tiles(temp,h)
    if temp!=0.
        beta=1/temp
        @show beta
        H=zeros(2,2,2,2)
        states=[]
        for s1=1:2,s2=1:2,s3=1:2,s4=1:2
            H[s1,s2,s3,s4]=0.5*(S(s1)*S(s2)+S(s2)*S(s3)+S(s3)*S(s4)+S(s4)*S(s1))+ S(s2)*S(s4) -h*(  1.0/6*(S(s1)+S(s3))+1.0/3*(S(s4)+S(s2))   )
            # print_square_tile((s1,s2,s3,s4))
            push!(states,(s1,s2,s3,s4))
        end
        E0 = minimum(H)
        m= [square_sym_mag(s) for s in states]
        op=[square_sym_order_parameter(s) for s in states]
        i=0
        for s in states
            i+=1
            if H[s...]==E0
                println("i= $i")
                print_square_tile(s)
            end

        end
        P=Array{Any}(undef,2)
        P[1]=zeros(Int64,length(states),length(states))
        P[2]=zeros(Int64,length(states),length(states))
        for (s1,i1) in zip(states,1:length(states)), (s2,i2) in zip(states,1:length(states))
            if s1[1]==s2[4]&&s1[2]==s2[3]
                P[1][i1,i2]=1
            end
            if s1[2]==s2[1]&&s1[3]==s2[4]
                P[2][i1,i2]=1
            end
        end
        F=Array{Any}(undef,2)
        U=Array{Any}(undef,4)
        indI=Array{Any}(undef,4)
        cutInd=Array{Any}(undef,4)
        for i=1:2
            F[i]=svd(P[i];full=true)
            # @show F[i].S
            cutInd[i]= findfirst(x->x<=1e-15,F[i].S)
            cutInd[i+2]= findfirst(x->x<=1e-15,F[i].S)
            if cutInd[i]==nothing
                cutInd[i]=length(F[i].S)+1
                cutInd[i+2]=length(F[i].S)+1
            end
            # @show F[i].S[cutInd[i]]
            U[i]=F[i].U[:,1:cutInd[i]-1]*Diagonal(map(x->sqrt(x),F[i].S[1:cutInd[i]-1]))
            U[i+2]=F[i].V[:,1:cutInd[i]-1]*Diagonal(map(x->sqrt(x),F[i].S[1:cutInd[i]-1]))
            indI[i]=Index(cutInd[i]-1,tags="i$i")
            indI[i+2]=Index(cutInd[i]-1,tags="i$(i+2)")
            # @show size(U[i])
        end
        A=ITensor(indI...)
        M=ITensor(indI...)
        E=ITensor(indI...)
        OP=ITensor(Complex{Float64},indI...)
        for k=1:length(states),i1=1:cutInd[1]-1,i2=1:cutInd[2]-1,i3=1:cutInd[3]-1,i4=1:cutInd[4]-1;
            A[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= exp(-beta*(H[states[k]...]-E0))*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            M[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= exp(-beta*(H[states[k]...]-E0))*m[k]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            OP[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= exp(-beta*(H[states[k]...]-E0))*op[k]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            E[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= exp(-beta*(H[states[k]...]-E0))*H[states[k]...]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
        end
        return (A,E,OP,M,U)
    else
        H=zeros(2,2,2,2)
        for s1=1:2,s2=1:2,s3=1:2,s4=1:2
            H[s1,s2,s3,s4]=0.5*(S(s1)*S(s2)+S(s2)*S(s3)+S(s3)*S(s4)+S(s4)*S(s1))+ S(s2)*S(s4) -h*(  1.0/6*(S(s1)+S(s3))+1.0/3*(S(s4)+S(s2))   )
        end
        E0 = minimum(H)
        @show E0
        gs=findall(x->x==E0,H)
        m= [square_sym_mag(s) for s in gs]
        op=[square_sym_order_parameter(s) for s in gs]
        i=0
        for s in gs
            i+=1
            println("i= $i")
        end
        @show length(gs)
        P=Array{Any}(undef,2)
        P[1]=zeros(Int64,length(gs),length(gs))
        P[2]=zeros(Int64,length(gs),length(gs))
        for (s1,i1) in zip(gs,1:length(gs)), (s2,i2) in zip(gs,1:length(gs))
            if s1[1]==s2[4]&&s1[2]==s2[3]
                P[1][i1,i2]=1
            end
            if s1[2]==s2[1]&&s1[3]==s2[4]
                P[2][i1,i2]=1
            end
        end
        F=Array{Any}(undef,2)
        U=Array{Any}(undef,4)
        indI=Array{Any}(undef,4)
        cutInd=Array{Any}(undef,4)
        for i=1:2
            F[i]=svd(P[i];full=true)
            cutInd[i]= findfirst(x->x<=1e-6,F[i].S)
            cutInd[i+2]= findfirst(x->x<=1e-6,F[i].S)
            if cutInd[i]==nothing
                cutInd[i]=length(F[i].S)+1
                cutInd[i+2]=length(F[i].S)+1
            end
            U[i]=F[i].U[:,1:cutInd[i]-1]*Diagonal(map(x->sqrt(x),F[i].S[1:cutInd[i]-1]))
            U[i+2]=F[i].V[:,1:cutInd[i]-1]*Diagonal(map(x->sqrt(x),F[i].S[1:cutInd[i]-1]))
            indI[i]=Index(cutInd[i]-1)
            indI[i+2]=Index(cutInd[i]-1)
            end
        A=ITensor(indI...)
        M=ITensor(indI...)
        M_single=[]
        for i=1:4
            push!(M_single,ITensor(indI...))
        end
        OP=ITensor(Complex{Float64},indI...)
        for k=1:length(gs),i1=1:cutInd[1]-1,i2=1:cutInd[2]-1,i3=1:cutInd[3]-1,i4=1:cutInd[4]-1;
            A[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            OP[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= op[k]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            M[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= m[k]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
        end
        return (A,E0*A,OP,M,U)
    end
end

function square_single_tiles(temp,h)
    if temp!=0.
        beta=1/temp
        @show beta
        H=zeros(2,2,2,2)
        states=[]
        for s1=1:2,s2=1:2,s3=1:2,s4=1:2
            H[s1,s2,s3,s4]=(S(s1)*S(s2)+S(s2)*S(s4)+S(s4)*S(s1))-h/3*(S(s1)+S(s2)+S(s4))
            push!(states,(s1,s2,s3,s4))
        end
        E0 = minimum(H)
        m= [square_single_mag(s) for s in states]
        op=[square_single_order_parameter(s) for s in states]
        i=0
        for s in states
            i+=1
            if H[s...]==E0
                println("i= $i")
                print_square_tile(s)
            end
        end
        P=Array{Any}(undef,2)
        P[1]=zeros(Int64,length(states),length(states))
        P[2]=zeros(Int64,length(states),length(states))
        for (s1,i1) in zip(states,1:length(states)), (s2,i2) in zip(states,1:length(states))
            if s1[1]==s2[4]&&s1[2]==s2[3]
                P[1][i1,i2]=1
            end
            if s1[2]==s2[1]&&s1[3]==s2[4]
                P[2][i1,i2]=1
            end
        end
        F=Array{Any}(undef,2)
        U=Array{Any}(undef,4)
        indI=Array{Any}(undef,4)
        cutInd=Array{Any}(undef,4)
        for i=1:2
            F[i]=svd(P[i];full=true)
            # @show F[i].S
            cutInd[i]= findfirst(x->x<=1e-15,F[i].S)
            cutInd[i+2]= findfirst(x->x<=1e-15,F[i].S)
            if cutInd[i]==nothing
                cutInd[i]=length(F[i].S)+1
                cutInd[i+2]=length(F[i].S)+1
            end
            # @show F[i].S[cutInd[i]]
            U[i]=F[i].U[:,1:cutInd[i]-1]*Diagonal(map(x->sqrt(x),F[i].S[1:cutInd[i]-1]))
            U[i+2]=F[i].V[:,1:cutInd[i]-1]*Diagonal(map(x->sqrt(x),F[i].S[1:cutInd[i]-1]))
            indI[i]=Index(cutInd[i]-1,tags="i$i")
            indI[i+2]=Index(cutInd[i]-1,tags="i$(i+2)")
            # @show size(U[i])
        end
        A=ITensor(indI...)
        M=ITensor(indI...)
        E=ITensor(indI...)
        OP=ITensor(Complex{Float64},indI...)
        for k=1:length(states),i1=1:cutInd[1]-1,i2=1:cutInd[2]-1,i3=1:cutInd[3]-1,i4=1:cutInd[4]-1;
            A[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= exp(-beta*(H[states[k]...]-E0))*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            M[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= exp(-beta*(H[states[k]...]-E0))*m[k]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            OP[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= exp(-beta*(H[states[k]...]-E0))*op[k]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            E[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= exp(-beta*(H[states[k]...]-E0))*H[states[k]...]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
        end
        return (A,E,OP,M,U)
    else
        H=zeros(2,2,2,2)
        for s1=1:2,s2=1:2,s3=1:2,s4=1:2
            H[s1,s2,s3,s4]=(S(s1)*S(s2)+S(s2)*S(s4)+S(s4)*S(s1))-h/3*(S(s1)+S(s2)+S(s4))
        end
        E0 = minimum(H)
        @show E0
        gs=findall(x->x==E0,H)
        m= [square_single_mag(s) for s in gs]
        op=[square_single_order_parameter(s) for s in gs]
        i=0
        for s in gs
            i+=1
            println("i= $i")
        end
        @show length(gs)
        P=Array{Any}(undef,2)
        P[1]=zeros(Int64,length(gs),length(gs))
        P[2]=zeros(Int64,length(gs),length(gs))
        for (s1,i1) in zip(gs,1:length(gs)), (s2,i2) in zip(gs,1:length(gs))
            if s1[1]==s2[4]&&s1[2]==s2[3]
                P[1][i1,i2]=1
            end
            if s1[2]==s2[1]&&s1[3]==s2[4]
                P[2][i1,i2]=1
            end
        end
        F=Array{Any}(undef,2)
        U=Array{Any}(undef,4)
        indI=Array{Any}(undef,4)
        cutInd=Array{Any}(undef,4)
        for i=1:2
            F[i]=svd(P[i];full=true)
            cutInd[i]= findfirst(x->x<=1e-6,F[i].S)
            cutInd[i+2]= findfirst(x->x<=1e-6,F[i].S)
            if cutInd[i]==nothing
                cutInd[i]=length(F[i].S)+1
                cutInd[i+2]=length(F[i].S)+1
            end
            U[i]=F[i].U[:,1:cutInd[i]-1]*Diagonal(map(x->sqrt(x),F[i].S[1:cutInd[i]-1]))
            U[i+2]=F[i].V[:,1:cutInd[i]-1]*Diagonal(map(x->sqrt(x),F[i].S[1:cutInd[i]-1]))
            indI[i]=Index(cutInd[i]-1)
            indI[i+2]=Index(cutInd[i]-1)
            end
        A=ITensor(indI...)
        M=ITensor(indI...)
        M_single=[]
        for i=1:4
            push!(M_single,ITensor(indI...))
        end
        OP=ITensor(Complex{Float64},indI...)
        for k=1:length(gs),i1=1:cutInd[1]-1,i2=1:cutInd[2]-1,i3=1:cutInd[3]-1,i4=1:cutInd[4]-1;
            A[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            OP[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= op[k]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
            M[indI[1]=>i1,indI[2]=>i2,indI[3]=>i3,indI[4]=>i4]+= m[k]*U[1][k,i1]*U[2][k,i2]*U[3][k,i3]*U[4][k,i4]
        end
        return (A,E0*A,OP,M,U)
    end
end

function hex_initialiser(temp,h,chi,boundaryCondition,InitialisationFile)
    (A,En,OP,M)=hex_tile(temp,h)
    # @show En
    alpha=[Index(4,tags="alpha$i") for i=1:4]
    beta=[Index(4,tags="beta$i") for i=1:4]
    if boundaryCondition==:Open
        E=Array{ITensor}(undef,4)
        E[1]=ITensor(Float64,alpha[1],alpha[2],inds(A)[1])
        E[2]=ITensor(Float64,beta[2],beta[3],inds(A)[2])
        E[3]=ITensor(Float64,alpha[3],alpha[4],inds(A)[3])
        E[4]=ITensor(Float64,beta[4],beta[1],inds(A)[4])
        C = [ITensor(Float64,alpha[i],beta[i]) for i=1:4]
        for a1=1:4,a2=1:4,b1=1:4,b2=1:4
            C[1][alpha[1]=>a1,beta[1]=>a2]+=A[b1,a1,a2,b2]
            C[2][beta[2]=>a1,alpha[2]=>a2]+=A[b1,b2,a1,a2]
            C[3][alpha[3]=>a1,beta[3]=>a2]+=A[a2,b1,b2,a1]
            C[4][beta[4]=>a1,alpha[4]=>a2]+=A[a1,a2,b1,b2]
        end
        for a1=1:4,a2=1:4,a3=1:4,b2=1:4
            E[1][alpha[2]=>a1,inds(A)[1]=>a2,alpha[1]=>a3]+=A[b2,a1,a2,a3]
            E[2][beta[3]=>a1,inds(A)[2]=>a2,beta[2]=>a3]  +=A[a3,b2,a1,a2]
            E[3][alpha[4]=>a1,inds(A)[3]=>a2,alpha[3]=>a3]+=A[a2,a3,b2,a1]
            E[4][beta[1]=>a1,inds(A)[4]=>a2,beta[4]=>a3]  +=A[a1,a2,a3,b2]
        end
    elseif boundaryCondition==:SymmetryBroken
        E=Array{ITensor}(undef,4)
        E[1]=ITensor(Float64,alpha[1],alpha[2],inds(A)[1])
        E[2]=ITensor(Float64,beta[2],beta[3],inds(A)[2])
        E[3]=ITensor(Float64,alpha[3],alpha[4],inds(A)[3])
        E[4]=ITensor(Float64,beta[4],beta[1],inds(A)[4])
        C = [ITensor(Float64,alpha[i],beta[i]) for i=1:4]
        for a1=1:4,a2=1:4,b1=1,b2=1
            C[1][alpha[1]=>a1,beta[1]=>a2]+=A[b2,a1,a2,b1]
            C[2][beta[2]=>a1,alpha[2]=>a2]+=A[b1,b2,a1,a2]
            C[3][alpha[3]=>a1,beta[3]=>a2]+=A[a2,b1,b2,a1]
            C[4][beta[4]=>a1,alpha[4]=>a2]+=A[a1,a2,b1,b2]
        end
        for a1=1:4,a2=1:4,a3=1:4,b2=1
            E[1][alpha[2]=>a1,inds(A)[1]=>a2,alpha[1]=>a3]+=A[b2,a1,a2,a3]
            E[2][beta[3]=>a1,inds(A)[2]=>a2,beta[2]=>a3]  +=A[a3,b2,a1,a2]
            E[3][alpha[4]=>a1,inds(A)[3]=>a2,alpha[3]=>a3]+=A[a2,a3,b2,a1]
            E[4][beta[1]=>a1,inds(A)[4]=>a2,beta[4]=>a3]  +=A[a1,a2,a3,b2]
        end
    elseif boundaryCondition==:RandomComplex
        E0=randn(Complex{Float64},(dim(alpha[1]),dim(alpha[1]),4))
        E=Array{ITensor}(undef,4)
        E[1]=ITensor(E0,alpha[1],alpha[2],inds(A)[1])
        E[2]=ITensor(E0,beta[2],beta[3],inds(A)[2])
        E[3]=ITensor(E0,alpha[3],alpha[4],inds(A)[3])
        E[4]=ITensor(E0,beta[4],beta[1],inds(A)[4])
        C0=randn(Complex{Float64},(dim(alpha[1]),dim(alpha[1])))
        C = [ITensor(C0+C0',alpha[i],beta[i]) for i=1:4]
    elseif boundaryCondition==:Load
        iNet=load_object(InitialisationFile)
        A=   replaceinds(A,inds(A),iNet.i)
        En=  replaceinds(En,inds(En),iNet.i)
        OP=  replaceinds(OP,inds(OP),iNet.i)
        M=   replaceinds(M,inds(M),iNet.i)
        E=iNet.E
        C=iNet.C
        alpha=iNet.alpha
        beta=iNet.beta
        @show inds(A)
    else
        error("Invalid boundary condition")
    end
    return (peps(E,C,chi,A,alpha,beta,inds(A),nothing),En,OP,M)
end

function square_initialiser(temp,h,chi,boundaryCondition,tensorConstruction,InitialisationFile)
    alpha=[Index(4,tags="alpha$i") for i=1:4]
    beta=[Index(4,tags="beta$i") for i=1:4]
    if tensorConstruction==:SquareSym
        (A,En,OP,M,U)=square_sym_tiles(temp,h)
    elseif tensorConstruction==:SquareSingle
        (A,En,OP,M,U)=square_single_tiles(temp,h)
    end
    if temp!=0.
        if boundaryCondition==:Open
            E=Array{ITensor}(undef,4)
            C=Array{ITensor}(undef,4)
            E[1]=ITensor(Float64,alpha[1],alpha[2],inds(A)[1])
            E[2]=ITensor(Float64,beta[2],beta[3],inds(A)[2])
            E[3]=ITensor(Float64,alpha[3],alpha[4],inds(A)[3])
            E[4]=ITensor(Float64,beta[4],beta[1],inds(A)[4])
            C = [ITensor(Float64,alpha[i],beta[i]) for i=1:4]
            for k=1:size(U[1],1)
                for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1])
                    C[1][alpha[1]=>a1,beta[1]=>a2]+=U[2][k,a1]*U[3][k,a2]
                    C[2][beta[2]=>a1,alpha[2]=>a2]+=U[3][k,a1]*U[4][k,a2]
                    C[3][beta[3]=>a1,alpha[3]=>a2]+=U[1][k,a1]*U[4][k,a2]
                    C[4][beta[4]=>a1,alpha[4]=>a2]+=U[1][k,a1]*U[2][k,a2]
                end
                for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1]),a3=1:dim(inds(A)[1])
                    E[1][alpha[2]=>a1,inds(A)[1]=>a2,alpha[1]=>a3]+=U[2][k,a1]*U[3][k,a2]*U[4][k,a3]
                    E[2][beta[3]=>a1,inds(A)[2]=>a2,beta[2]=>a3]  +=U[3][k,a1]*U[4][k,a2]*U[1][k,a3]
                    E[3][alpha[4]=>a1,inds(A)[3]=>a2,alpha[3]=>a3]+=U[4][k,a1]*U[1][k,a2]*U[2][k,a3]
                    E[4][beta[1]=>a1,inds(A)[4]=>a2,beta[4]=>a3]  +=U[1][k,a1]*U[2][k,a2]*U[3][k,a3]
                end
            end
        elseif boundaryCondition==:SymmetryBroken
            println("Correct branch")
            E=Array{ITensor}(undef,4)
            C=Array{ITensor}(undef,4)
            E[1]=ITensor(Float64,alpha[1],alpha[2],inds(A)[1])
            E[2]=ITensor(Float64,beta[2],beta[3],inds(A)[2])
            E[3]=ITensor(Float64,alpha[3],alpha[4],inds(A)[3])
            E[4]=ITensor(Float64,beta[4],beta[1],inds(A)[4])
            C = [ITensor(Float64,alpha[i],beta[i]) for i=1:4]
            if tensorConstruction==:SquareSym||true
                vk=[2,5,11]
            end
            for k in vk
                for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1]),a3=1:dim(inds(A)[1])
                    E[1][alpha[2]=>a1,inds(A)[1]=>a2,alpha[1]=>a3]+=U[2][k,a1]*U[3][k,a2]*U[4][k,a3]
                    E[2][beta[3]=>a1,inds(A)[2]=>a2,beta[2]=>a3]  +=U[3][k,a1]*U[4][k,a2]*U[1][k,a3]
                    E[3][alpha[4]=>a1,inds(A)[3]=>a2,alpha[3]=>a3]+=U[4][k,a1]*U[1][k,a2]*U[2][k,a3]
                    E[4][beta[1]=>a1,inds(A)[4]=>a2,beta[4]=>a3]  +=U[1][k,a1]*U[2][k,a2]*U[3][k,a3]
                end
                for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1])
                    C[2][beta[2]=>a1,alpha[2]=>a2]+=U[3][k,a1]*U[4][k,a2]
                    C[3][beta[3]=>a1,alpha[3]=>a2]+=U[1][k,a1]*U[4][k,a2]
                    C[4][beta[4]=>a1,alpha[4]=>a2]+=U[1][k,a1]*U[2][k,a2]
                end
            end
            for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1])
                k=2
                C[1][alpha[1]=>a1,beta[1]=>a2]+=U[2][k,a1]*U[3][k,a2]
            end
        elseif boundaryCondition==:RandomComplex
            E0=randn(Complex{Float64},(dim(alpha[1]),dim(alpha[1]),4))
            E=Array{ITensor}(undef,4)
            E[1]=ITensor(E0,alpha[1],alpha[2],inds(A)[1])
            E[2]=ITensor(E0,beta[2],beta[3],inds(A)[2])
            E[3]=ITensor(E0,alpha[3],alpha[4],inds(A)[3])
            E[4]=ITensor(E0,beta[4],beta[1],inds(A)[4])
            C0=randn(Complex{Float64},(dim(alpha[1]),dim(alpha[1])))
            C = [ITensor(C0+C0',alpha[i],beta[i]) for i=1:4]
        end
    else
        if boundaryCondition==:Open
            E=Array{ITensor}(undef,4)
            C=Array{ITensor}(undef,4)
            E[1]=ITensor(Float64,alpha[1],alpha[2],inds(A)[1])
            E[2]=ITensor(Float64,beta[2],beta[3],inds(A)[2])
            E[3]=ITensor(Float64,alpha[3],alpha[4],inds(A)[3])
            E[4]=ITensor(Float64,beta[4],beta[1],inds(A)[4])
            C = [ITensor(Float64,alpha[i],beta[i]) for i=1:4]
            for k=1:size(U[1],1)
                for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1])
                    C[1][alpha[1]=>a1,beta[1]=>a2]+=U[2][k,a1]*U[3][k,a2]
                    C[2][beta[2]=>a1,alpha[2]=>a2]+=U[3][k,a1]*U[4][k,a2]
                    C[3][beta[3]=>a1,alpha[3]=>a2]+=U[1][k,a1]*U[4][k,a2]
                    C[4][beta[4]=>a1,alpha[4]=>a2]+=U[1][k,a1]*U[2][k,a2]
                end
                for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1]),a3=1:dim(inds(A)[1])
                    E[1][alpha[2]=>a1,inds(A)[1]=>a2,alpha[1]=>a3]+=U[2][k,a1]*U[3][k,a2]*U[4][k,a3]
                    E[2][beta[3]=>a1,inds(A)[2]=>a2,beta[2]=>a3]  +=U[3][k,a1]*U[4][k,a2]*U[1][k,a3]
                    E[3][alpha[4]=>a1,inds(A)[3]=>a2,alpha[3]=>a3]+=U[4][k,a1]*U[1][k,a2]*U[2][k,a3]
                    E[4][beta[1]=>a1,inds(A)[4]=>a2,beta[4]=>a3]  +=U[1][k,a1]*U[2][k,a2]*U[3][k,a3]
                end
            end
        elseif boundaryCondition==:SymmetryBroken
            E=Array{ITensor}(undef,4)
            C=Array{ITensor}(undef,4)
            E[1]=ITensor(Float64,alpha[1],alpha[2],inds(A)[1])
            E[2]=ITensor(Float64,beta[2],beta[3],inds(A)[2])
            E[3]=ITensor(Float64,alpha[3],alpha[4],inds(A)[3])
            E[4]=ITensor(Float64,beta[4],beta[1],inds(A)[4])
            C = [ITensor(Float64,alpha[i],beta[i]) for i=1:4]
            for k in [5,8,10]
                for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1]),a3=1:dim(inds(A)[1])
                    E[1][alpha[2]=>a1,inds(A)[1]=>a2,alpha[1]=>a3]+=U[2][k,a1]*U[3][k,a2]*U[4][k,a3]
                    E[2][beta[3]=>a1,inds(A)[2]=>a2,beta[2]=>a3]  +=U[3][k,a1]*U[4][k,a2]*U[1][k,a3]
                    E[3][alpha[4]=>a1,inds(A)[3]=>a2,alpha[3]=>a3]+=U[4][k,a1]*U[1][k,a2]*U[2][k,a3]
                    E[4][beta[1]=>a1,inds(A)[4]=>a2,beta[4]=>a3]  +=U[1][k,a1]*U[2][k,a2]*U[3][k,a3]
                end
                for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1])
                    C[2][beta[2]=>a1,alpha[2]=>a2]+=U[3][k,a1]*U[4][k,a2]
                    C[3][beta[3]=>a1,alpha[3]=>a2]+=U[1][k,a1]*U[4][k,a2]
                    C[4][beta[4]=>a1,alpha[4]=>a2]+=U[1][k,a1]*U[2][k,a2]
                end
            end
            for a1=1:dim(inds(A)[1]),a2=1:dim(inds(A)[1])
                k=5
                C[1][alpha[1]=>a1,beta[1]=>a2]+=U[2][k,a1]*U[3][k,a2]
            end
        elseif boundaryCondition==:RandomComplex
            E0=randn(Complex{Float64},(dim(alpha[1]),dim(alpha[1]),4))
            E=Array{ITensor}(undef,4)
            E[1]=ITensor(E0,alpha[1],alpha[2],inds(A)[1])
            E[2]=ITensor(E0,beta[2],beta[3],inds(A)[2])
            E[3]=ITensor(E0,alpha[3],alpha[4],inds(A)[3])
            E[4]=ITensor(E0,beta[4],beta[1],inds(A)[4])
            C0=randn(Complex{Float64},(dim(alpha[1]),dim(alpha[1])))
            C = [ITensor(C0+C0',alpha[i],beta[i]) for i=1:4]
        elseif boundaryCondition==:Load
            iNet=load_object(InitialisationFile)
            A=   replaceinds(A,inds(A),iNet.i)
            En=  replaceinds(En,inds(En),iNet.i)
            OP=  replaceinds(OP,inds(OP),iNet.i)
            M=   replaceinds(M,inds(M),iNet.i)
            E=iNet.E
            C=iNet.C
            alpha=iNet.alpha
            beta=iNet.beta
            @show inds(A)
        else
            error("Invalid boundary condition")
        end
    end

    return (peps(E,C,chi,A,alpha,beta,inds(A),nothing),En,OP,M)
end

function triangle_initialiser(temp,h,chi,boundaryCondition,tensorConstruction,InitialisationFile)
    if tensorConstruction==:Hex
        return hex_initialiser(temp,h,chi,boundaryCondition,InitialisationFile)
    elseif tensorConstruction==:SquareSym||tensorConstruction==:SquareSingle
        return square_initialiser(temp,h,chi,boundaryCondition,tensorConstruction,InitialisationFile);
    end
end

function ssMag(curNet,temp,h,tensorConstruction)
    if tensorConstruction==:Hex
        i=inds(curNet.A)
        vMag=[ITensor(i...) for k=0:6]
        if h<=6
            e0=-3-h
        else
            e0=9-3*h
        end
        i=0
        for s0=1:2,s1=1:2,s2=1:2,s3=1:2,s4=1:2,s5=1:2,s6=1:2
            if temp!=0
                vMag[1][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s1)*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)
                vMag[2][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s2)*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)
                vMag[3][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s3)*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)
                vMag[4][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s4)*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)
                vMag[5][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s5)*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)
                vMag[6][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s6)*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)
                vMag[7][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s0)*exp(-hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)/temp)
            elseif temp==0
                if (hex_energy(s0,[s1,s2,s3,s4,s5,s6],h)-e0)<=1e-7
                    i=i+1
                    vMag[1][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s1)
                    vMag[2][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s2)
                    vMag[3][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s3)
                    vMag[4][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s4)
                    vMag[5][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s5)
                    vMag[6][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s6)
                    vMag[7][bundle(s6,s1),bundle(s1,s2),bundle(s4,s3),bundle(s5,s4)]+=S(s0)
                end
            end
        end
        if temp!=0
            lambda=exp(-e0/temp)
        else
            lambda=1
        end
        for k=1:7
            vMag[k] = vMag[k]/lambda
        end
        # lambda=maximum(A)
        return vMag
    else
        return nothing;
    end
end