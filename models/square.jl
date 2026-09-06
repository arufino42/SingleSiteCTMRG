
function S(a)
    return 3-2*a
end

function delta(i1::Int,i2::Int)
    return i1==i2 ? 1 : 0
end
function exactIsingM(T::Number)
    beta=1/T
    betac=log(1+sqrt(2))/2
    return (beta>betac) ? (1-(sinh(2*beta))^(-4))^(1/8) : 0
end

function exactIsingCorr(beta::Number)
    k=1/(sinh(2*beta)*sinh(2*beta))
    return k<1 ? -1/(log(k))/sqrt(2) : 2/log(k)/sqrt(2)
end

function exactIsingF(T)
    beta=1/T
    k=1/sinh(2*beta)^2
    f(x) =log(cosh(2*beta)^2 +1/k*sqrt(1+k^2-2*k*cos(2*x)))
    I,Err=quadgk(
    f,0,pi;
    rtol = 1e-6
    )
    # @show I
    return log(2)/2+1/(2*pi)*I-2/T
end

function isingSquareTensors(T::Number,h::Number,tensorConstruction::Symbol)
    if tensorConstruction==:SingleSite
        if T>0
            Q=[1 exp((-2 - h/2)/T);exp((-2-h/2)/T) exp(-h/T)]
            P=sqrt(Q)
        elseif T<0
            Q=[exp((-2+h/2)/abs(T)) 1;1 exp((-2-h/2)/abs(T))]
            P=sqrt(Q)
        elseif T === +0.
            Q=[1 0;0 0]
            P=sqrt(Q)
        elseif T === -0.
            if h<4
                Q=[0 1;1 0]
            elseif h==4.
                Q=[1 1;1 0]
            elseif h>4.
                Q=[1 0;0 0]
            end
            P=sqrt(Q)
        end
        i1=Index(2;tags="i1");
        i2=Index(2;tags="i2");
        i3=Index(2;tags="i3");
        i4=Index(2;tags="i4");
        A=ITensor(Complex{Float64},i1,i2,i3,i4);
        M=ITensor(Complex{Float64},i1,i2,i3,i4);
        E=ITensor(Complex{Float64},i1,i2,i3,i4);
        e = sign(T) * [-1 1;1 -1]
        X=e*(P^-1)
        for j1=1:2,j2=1:2,j3=1:2,j4=1:2,i=1:2
            A[j1,j2,j3,j4]+= P[i,j1]*P[i,j2]*P[j3,i]*P[j4,i]
            M[j1,j2,j3,j4]+= S(i)*P[i,j1]*P[i,j2]*P[j3,i]*P[j4,i]
            E[j1,j2,j3,j4]+= P[i,j1]*X[i,j2]*P[j3,i]*P[j4,i]
        end
    elseif tensorConstruction==:Plaquette
        i1=Index(2;tags="i1");
        i2=Index(2;tags="i2");
        i3=Index(2;tags="i3");
        i4=Index(2;tags="i4");
        A=ITensor(Complex{Float64},i1,i2,i3,i4);
        M=ITensor(Complex{Float64},i1,i2,i3,i4);
        E=ITensor(Complex{Float64},i1,i2,i3,i4);
        if T!=0
            for a1=1:2,a2=1:2,a3=1:2,a4=1:2
                A[a1,a2,a3,a4]= exp( (S(a1)*S(a2)+S(a2)*S(a3)+S(a3)*S(a4)+S(a4)*S(a1))/T +0.5*h/abs(T)*(S(a1)+S(a2)+S(a3)+S(a4)))
                M[a1,a2,a3,a4]= 0.25*(S(a1)+sign(T)*S(a2)+S(a3)+sign(T)*S(a4))*exp( (S(a1)*S(a2)+S(a2)*S(a3)+S(a3)*S(a4)+S(a4)*S(a1))/T +0.5*h/abs(T)*(S(a1)+S(a2)+S(a3)+S(a4)))
                E[a1,a2,a3,a4]=( sign(T)*0.25*(S(a1)*S(a2)+S(a2)*S(a3)+S(a3)*S(a4)+S(a4)*S(a1)))*exp( (S(a1)*S(a2)+S(a2)*S(a3)+S(a3)*S(a4)+S(a4)*S(a1))/T +0.5*h/abs(T)*(S(a1)+S(a2)+S(a3)+S(a4)))
            end
            lambda=maximum(abs.(array(A)))
            A=A/lambda
            M=M/lambda
            E=E/lambda
        elseif T==0
            # for a1=1:2,a2=1:2,a3=1:2,a4=1:2
            #     A[a1,a2,a3,a4]= (-(S(a1)*S(a2)+S(a2)*S(a3)+S(a3)*S(a4)+S(a4)*S(a1)) +0.5*h*(S(a1)+S(a2)+S(a3)+S(a4))-E0(h))>=0 ? 1 : 0
            # end
        end
        P=nothing
    end
    return (A,E,M,P)
end

function initialiseCTMRG_square(chi::Int,T::Number,h::Number,boundaryCondition::Symbol,tensorConstruction::Symbol)
    (A,En,M,P)=isingSquareTensors(T,h,tensorConstruction)
    if boundaryCondition==:Open
        if tensorConstruction==:SingleSite
            alpha=[Index(2,tags="alpha$i") for i=1:4]
            beta=[Index(2,tags="beta$i") for i=1:4]
            E=Array{ITensor}(undef,4)
            E[1]=ITensor(Complex{Float64},alpha[1],alpha[2],inds(A)[1])
            E[2]=ITensor(Complex{Float64},beta[2],beta[3],inds(A)[2])
            E[3]=ITensor(Complex{Float64},alpha[3],alpha[4],inds(A)[3])
            E[4]=ITensor(Complex{Float64},beta[4],beta[1],inds(A)[4])
            C = [ITensor(Complex{Float64},alpha[i],beta[i]) for i=1:4]
            for a1=1:2,a2=1:2,i=1:2
                C[1][alpha[1]=>a1,beta[1]=>a2]+=P[a1,i]*P[i,a2]
                C[2][beta[2]=>a1,alpha[2]=>a2]+=P[a1,i]*P[a2,i]
                C[3][alpha[3]=>a1,beta[3]=>a2]+=P[i,a1]*P[a2,i]
                C[4][beta[4]=>a1,alpha[4]=>a2]+=P[i,a1]*P[i,a2]
            end
            for a1=1:2,a2=1:2,a3=1:2,i=1:2
                E[1][alpha[2]=>a1,inds(A)[1]=>a2,alpha[1]=>a3]+=P[i,a1]*P[a1,i]*P[a1,i]
                E[2][beta[3]=>a1,inds(A)[2]=>a2,beta[2]=>a3]  +=P[a1,i]*P[a1,i]*P[i,a1]
                E[3][alpha[4]=>a1,inds(A)[3]=>a2,alpha[3]=>a3]+=P[a1,i]*P[i,a1]*P[i,a1]
                E[4][beta[1]=>a1,inds(A)[4]=>a2,beta[4]=>a3]  +=P[i,a1]*P[i,a1]*P[a1,i]
            end
        elseif tensorConstruction==:Plaquette
            alpha=[Index(2,tags="alpha$i") for i=1:4]
            beta=[Index(2,tags="beta$i") for i=1:4]
            E=Array{ITensor}(undef,4)
            E[1]=ITensor(alpha[1],alpha[2],inds(A)[1])
            E[2]=ITensor(beta[2],beta[3],inds(A)[2])
            E[3]=ITensor(alpha[3],alpha[4],inds(A)[3])
            E[4]=ITensor(beta[4],beta[1],inds(A)[4])
            C = [ITensor(alpha[i],beta[i]) for i=1:4]
            for i=1:4
                for a1=1:2,a2=1:2,i1=1:2,i2=1:2
                    C[i][a1,a2]+=A[a1,a2,i1,i2]
                end
                for a1=1:2,a2=1:2,a3=1:2,i1=1:2
                    E[i][a1,a2,a3]+=A[a1,a2,a3,i1]
                end
            end
        end
    elseif boundaryCondition==:RandomComplex
        alpha=[Index(2,tags="alpha$i") for i=1:4]
        beta=[Index(2,tags="beta$i") for i=1:4]
        E0=randn(Complex{Float64},(dim(alpha[1]),dim(alpha[1]),2))
        E=Array{ITensor}(undef,4)
        E[1]=ITensor(E0,alpha[1],alpha[2],inds(A)[1])
        E[2]=ITensor(E0,beta[2],beta[3],inds(A)[2])
        E[3]=ITensor(E0,alpha[3],alpha[4],inds(A)[3])
        E[4]=ITensor(E0,beta[4],beta[1],inds(A)[4])
        C0=randn(Complex{Float64},(dim(alpha[1]),dim(alpha[1])))
        C = [ITensor(C0+C0',alpha[i],beta[i]) for i=1:4]
    elseif boundaryCondition==:SymmetryBroken
        if tensorConstruction==:SingleSite
            alpha=[Index(2,tags="alpha$i") for i=1:4]
            beta=[Index(2,tags="beta$i") for i=1:4]
            E=Array{ITensor}(undef,4)
            E[1]=ITensor(Complex{Float64},alpha[1],alpha[2],inds(A)[1])
            E[2]=ITensor(Complex{Float64},beta[2],beta[3],inds(A)[2])
            E[3]=ITensor(Complex{Float64},alpha[3],alpha[4],inds(A)[3])
            E[4]=ITensor(Complex{Float64},beta[4],beta[1],inds(A)[4])
            C = [ITensor(Complex{Float64},alpha[i],beta[i]) for i=1:4]
            Y =[0 1;1 0]
            if T>0
                i=1
                for a1=1:2,a2=1:2
                    C[1][alpha[1]=>a1,beta[1]=>a2]+=P[a1,i]*P[i,a2]
                    C[2][beta[2]=>a1,alpha[2]=>a2]+=P[a1,i]*P[a2,i]
                    C[3][alpha[3]=>a1,beta[3]=>a2]+=P[i,a1]*P[a2,i]
                    C[4][beta[4]=>a1,alpha[4]=>a2]+=P[i,a1]*P[i,a2]
                end
                for a1=1:2,a2=1:2,a3=1:2
                    E[1][alpha[2]=>a1,inds(A)[1]=>a2,alpha[1]=>a3]+=P[i,a1]*P[a2,i]*P[a3,i]
                    E[2][beta[3]=>a1,inds(A)[2]=>a2,beta[2]=>a3]  +=P[a1,i]*P[a2,i]*P[i,a3]
                    E[3][alpha[4]=>a1,inds(A)[3]=>a2,alpha[3]=>a3]+=P[a1,i]*P[i,a2]*P[i,a3]
                    E[4][beta[1]=>a1,inds(A)[4]=>a2,beta[4]=>a3]  +=P[i,a1]*P[i,a2]*P[a3,i]
                end
            else
                for a1=1:2,a2=1:2,i=1
                    C[1][alpha[1]=>a1,beta[1]=>a2]+=P[a1,i]*P[i,a2]
                    C[2][beta[2]=>a1,alpha[2]=>a2]+=P[a1,i]*P[a2,i]
                    C[3][alpha[3]=>a1,beta[3]=>a2]+=P[i,a1]*P[a2,i]
                    C[4][beta[4]=>a1,alpha[4]=>a2]+=P[i,a1]*P[i,a2]
                end
                for a1=1:2,a2=1:2,a3=1:2,i=1:2
                    E[1][alpha[2]=>a1,inds(A)[1]=>a2,alpha[1]=>a3]+=P[i,a1]*P[a3,i]*P[a2,i]
                    E[2][beta[3]=>a1,inds(A)[2]=>a2,beta[2]=>a3]  +=P[a1,i]*P[i,a3]*P[a2,i]
                    E[3][alpha[4]=>a1,inds(A)[3]=>a2,alpha[3]=>a3]+=P[a1,i]*P[i,a3]*P[i,a2]
                    E[4][beta[1]=>a1,inds(A)[4]=>a2,beta[4]=>a3]  +=P[i,a1]*P[a3,i]*P[i,a2]
                end
            end
        elseif tensorConstruction==:Plaquette
            if T>0
                alpha=[Index(2,tags="alpha$i") for i=1:4]
                beta=[Index(2,tags="beta$i") for i=1:4]
                E=Array{ITensor}(undef,4)
                E[1]=ITensor(alpha[1],alpha[2],inds(A)[1])
                E[2]=ITensor(beta[2],beta[3],inds(A)[2])
                E[3]=ITensor(alpha[3],alpha[4],inds(A)[3])
                E[4]=ITensor(beta[4],beta[1],inds(A)[4])
                C = [ITensor(alpha[i],beta[i]) for i=1:4]
                for i=1:4
                    for a1=1:2,a2=1:2
                        C[i][a1,a2]=A[a1,a2,1,1]
                    end
                    for a1=1:2,a2=1:2,a3=1:2
                        E[i][a1,a2,a3]=A[a1,a2,a3,1]
                    end
                end
            elseif T<0
                alpha=[Index(2,tags="alpha$i") for i=1:4]
                beta=[Index(2,tags="beta$i") for i=1:4]
                E=Array{ITensor}(undef,4)
                E[1]=ITensor(alpha[1],alpha[2],inds(A)[1])
                E[2]=ITensor(beta[2],beta[3],inds(A)[2])
                E[3]=ITensor(alpha[3],alpha[4],inds(A)[3])
                E[4]=ITensor(beta[4],beta[1],inds(A)[4])
                C = [ITensor(alpha[i],beta[i]) for i=1:4]

                for a1=1:2,a2=1:2
                    C[1][a1,a2]=A[a1,a2,1,2]
                    C[2][a1,a2]=A[a1,a2,1,2]
                    C[3][a1,a2]=A[a1,a2,1,2]
                    C[4][a1,a2]=A[a1,a2,1,2]
                end
                for a1=1:2,a2=1:2,a3=1:2
                    E[1][a1,a2,a3]=A[a1,a2,a3,1]
                    E[2][a1,a2,a3]=A[a1,a2,a3,2]
                    E[3][a1,a2,a3]=A[a1,a2,a3,1]
                    E[4][a1,a2,a3]=A[a1,a2,a3,2]
                end
            end
        end

    end
    return (peps(E,C,chi,A,alpha,beta,inds(A),nothing),En,M)
end

function ferro_energy(net::peps,M)
    gamma1=Index(dim(net.alpha[1]))
    gamma2=Index(dim(net.i[2]))
    gamma3=Index(dim(net.alpha[3]))
    k1=Index(dim(net.i[2]))
    k3=Index(dim(net.i[2]))
    El1 = replaceind(net.E[1],net.alpha[2],gamma1)
    El3 = replaceind(net.E[3],net.alpha[3],gamma3)
    Er1 = replaceinds(net.E[1],[net.alpha[1],net.i[1]],[gamma1,k1])
    Er3 = replaceinds(net.E[3],[net.alpha[4],net.i[3]],[gamma3,k3])

    Ml = replaceind(M,net.i[2],gamma2)
    Mr = replaceinds(M,[net.i[1],net.i[4],net.i[3]],[k1,gamma2,k3])

    Al = replaceind(net.A,net.i[2],gamma2)
    Ar = replaceinds(net.A,[net.i[1],net.i[4],net.i[3]],[k1,gamma2,k3])

    env2=net.C[1]*El1*Er1*net.C[2]*net.E[2]*net.C[3]*Er3*El3*net.C[4]*net.E[4]
    # @show esnv2.inds
    z= (env2*Al*Ar)[]
    return (-(env2*Ml*Mr)[]/z)
end
a=0

function create_mStag(T::Number,h::Number,i)
    if T>=0
        Q=[1 exp((-2 - h/2)/T);exp((-2-h/2)/T) exp(-h/T)]
        P=sqrt(Q)
    elseif T<0
        Q=[exp((-2+h/2)/abs(T)) 1;1 exp((-2-h/2)/abs(T))]
        P=sqrt(Q)
    end
    k1=Index(2;tags="k1")
    k3=Index(2;tags="k3")
    Mstag=ITensor(Complex{Float64},i[1],k1,i[2],k3,i[3],i[4])
    for b1=1:2,b2=1:2,a1=1:2,a2=1:2,a3=1:2,a4=1:2,a5=1:2,a6=1:2
        Mstag[a1,a2,a3,a4,a5,a6]+=(S(b1)+S(b2))/2*Q[b1,b2]*P[b1,a1]*P[b2,a2]*P[b2,a3]*P[b2,a4]*P[b1,a5]*P[b1,a6]
    end
    return (Mstag,k1,k3)
end
