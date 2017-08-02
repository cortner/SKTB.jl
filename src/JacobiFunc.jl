module JacobiFunc

# Jacobi elliptic functions and complete elliptic integral of the first kind.
#
# References:
#      iK: Abramowitz, Stegun: Handbook of Mathematical Functions, Sec. 17.6.
#  jacobi: Abramowitz, Stegun: Handbook of Mathematical Functions, Sec. 16.12 & 16.13.
#          ellipjc.m in Toby Driscoll's Schwarz-Christoffel Toolbox
#

function iK(m::AbstractFloat)
    a,b = one(m),sqrt(m)
    while abs(a-b) > eps(a+b)
        a,b = 0.5*(a+b),sqrt(a*b)
    end
    return π/(2*a)
end
iK(m::Integer) = iK(float(m))
K(m) = iK(one(m)-m)

function landaudescend(m::AbstractFloat)
    f = one(m)
    κ = Vector{typeof(m)}()
    while true 
        if m > 1e-3
            push!(κ, (1-sqrt(1-m))/(1+sqrt(1-m)))
        else
            # Improve accuracy by evaluating a Taylor expansion 
            # instead of original formula for small m
            push!(κ, @evalpoly m/4 0 1 2 5 14 42 132)
        end
        m = κ[end]^2
        f *= 1+κ[end]
        if m < 4*eps(typeof(m)) break end
    end
    return f,κ
end
landaudescend(m::Integer) = landaudescend(float(m))

function jacobiκ(u::Number,κ)
    sn,cn,dn = sin(u) - κ[end]^2/4 * (u - sin(u)*cos(u)) * cos(u),
               cos(u) - κ[end]^2/4 * (u - sin(u)*cos(u)) * sin(u),
               1 - κ[end]^2/2 * sin(u)^2;
    for i = length(κ):-1:1
        sn,cn,dn = (1+κ[i])*sn/(1+κ[i]*sn^2),
                         cn*dn/(1+κ[i]*sn^2),
                 (1-κ[i]*sn^2)/(1+κ[i]*sn^2)
    end
    return sn,cn,dn
end
jacobiκ(u::AbstractArray, κ) = [jacobiκ(u,κ) for u in u]

function jacobi(u,m)
    f,κ = landaudescend(m)
    return jacobiκ(u/f,κ)
end

end # module
