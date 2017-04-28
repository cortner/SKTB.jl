module FermiContour

module Transforms

include("JacobiFunc.jl")

import Base.(|>)

immutable Sn{M}
    m::M
end
@inline function |>(p, s::Sn)
    w,z = p
    sn,cn,dn = JacobiFunc.jacobi(z, s.m)
    return w*cn*dn, sn
end

immutable Möbius{A,B,C,D}
    a::A; b::B; c::C; d::D
end
@inline function |>(p, m::Möbius)
    w,z = p
    return w*(m.a*(m.c*z+m.d ) - m.c*(m.a*z+m.b))/(m.c*z+m.d)^2,
          (m.a*z+m.b)/(m.c*z+m.d)
end

immutable Affine{A,B}
    a::A; b::B
end
@inline function |>(p, a::Affine)
    w,z = p
    return w*a.a, a.a*z+a.b
end

immutable Sqrt end
@inline function |>(p, s::Sqrt)
    w,z = p
    return w/(2*sqrt(z)), sqrt(z)
end

end # module Transforms


include("JacobiFunc.jl")

export fermicontour
function fermicontour(Emin,Emax,β,μ,n)
    m = Emin^2 + π^2/β^2
    M = Emax^2 + π^2/β^2

    k = (sqrt(M/m)-1)/(sqrt(M/m)+1)
    K = JacobiFunc.K(k^2)
    iK = im*JacobiFunc.iK(k^2)

    t = -K + iK/2 + 4*K*(0.5:1:2*n-0.5)/(2*n)
    w = Vector{Complex{typeof(k)}}(length(t))
    z = Vector{Complex{typeof(k)}}(length(t))
    for i = 1:length(t)
        # Quadrature points and weights
        w[i],z[i] = (1/(2π*im) * 4*K/n, t[i]) |>
            Transforms.Sn(k^2) |>
            Transforms.Möbius(1,1/k, -1,1/k) |>
            Transforms.Affine(-sqrt(m*M), π^2/β^2) |>
            Transforms.Sqrt() |>
            Transforms.Affine(im,0)
        z[i] += μ
    end
    return w,z
end

export fermi
function fermi(H, Emin,Emax,β,μ,n)
    w,z = fermicontour(Emin,Emax,β,μ,n)
    for i = 1:length(w)
       w[i] *= abs(β*z[i]) < log(realmax(typeof(k))) ?
               1/(1 + exp(β*z[i])) :
               0.5*(1-sign(real(z[i])))
    end
    return sum([real(w*inv(H-z*eye(H))) for (w,z) in zip(w,z)])
end

end # module
