using TightBinding
using JuLIP, JuLIP.ASE
using JuLIP.Testing
using Base.Test

COMPAREATOMS = false    # if Atoms.jl is installed
COMPAREQUIP = false     # if QUIP and quippy are installed
TESTDEPTH = 1

println("============================================")
println("    TightBinding Tests  ")
println("============================================")

# =========== Main tests =================
include("testtoymodel.jl")
include("testnrltb.jl")
include("testsiteE.jl")
include("testkwon.jl")
# include("test0T.jl")

# ============= Compare with Atoms.jl and QUIP implementations
if COMPAREATOMS; include("compareatoms.jl"); end
if COMPAREQUIP; include("comparequip.jl"); end

# ============= Performance benchmarks
# (uncomment these only for performance tests)
# include("benchmarkEandFrc.jl")
# include("perfsiteE.jl")


# =========== TEMP =======

# TODO: add the test for the fermi-contour
# using FermiContour
# using Base.Test
#
# using PyPlot
#
# # Compare the rate of convergence with the predicted rate.
# # Not a proper unit test, but good enough for our purposes.
#
# E1 = 0
# E2 = 1
# β = 1.0
# n = 1:20
# x = linspace(E1,E2,100)
#
# # Compute empirical convergence rate
# error = zeros(length(n))
# for i = 1:length(n)
#     w,z = fermicontour(E1,E2,β,n[i])
#     fx = sum(real(w*fermidirac(z,β)./(z-x)) for (w,z) in zip(w,z))
#     error[i] = maximum(abs.(fermidirac.(x,β) .- fx))
# end
#
# # Compute theoretical convergence rate
# m = E1^2 + π^2/β^2
# M = E2^2 + π^2/β^2
# k = (sqrt(M/m)-1)/(sqrt(M/m)+1)
# K = FermiContour.JacobiFunc.K(k^2)
# iK = FermiContour.JacobiFunc.iK(k^2)
#
#
# clf()
# semilogy(n,error,"-x")
# semilogy(n,exp(-π*iK/(2K)*n))
# ylim([1e-18,1])
# show()
