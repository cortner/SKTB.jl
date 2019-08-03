using Test
using SKTB, JuLIP, LinearAlgebra
using JuLIP.Testing

COMPAREQUIP = false     # if QUIP and quippy are installed
TESTDEPTH = 1

println("============================================")
println("    SKTB Tests  ")
println("============================================")

# =========== Main tests =================
@testset "SKTB Tests" begin
   include("testtoymodel.jl")
   include("testnrltb.jl")
   include("testcontour.jl")
   include("testsiteE.jl")
   include("testkwon.jl")
   include("testdual.jl")
   include("test0T.jl")
   include("testvirial.jl")
end

# ============= Compare with Atoms.jl and QUIP implementations
if COMPAREQUIP; include("comparequip.jl"); end

# ============= Performance benchmarks
# (uncomment these only for performance tests)
# include("benchmarkEandFrc.jl")
# include("perfsiteE.jl")


# =========== TEMP =======
