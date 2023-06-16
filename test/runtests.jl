using Test
using SKTB, JuLIP, LinearAlgebra
using JuLIP.Testing

COMPAREQUIP = false     # if QUIP and quippy are installed
TESTDEPTH = 1

##

println("============================================")
println("    SKTB Tests  ")
println("============================================")

# =========== Main tests =================
@testset "SKTB Tests" begin
   @testset "Toy Model" begin include("testtoymodel.jl"); end 
   @testset "NRLTB" begin include("testnrltb.jl"); end 
   @testset "Contour" begin include("testcontour.jl"); end 
   @testset "Site Energy" begin include("testsiteE.jl"); end 
   @testset "Kwon" begin include("testkwon.jl"); end 
   @testset "Dual" begin include("testdual.jl"); end 
   @testset "Zero-T" begin include("test0T.jl"); end 
   @testset "Virial" begin include("testvirial.jl"); end 
end

# ============= Compare with Atoms.jl and QUIP implementations
if COMPAREQUIP; include("comparequip.jl"); end

# ============= Performance benchmarks
# (uncomment these only for performance tests)
# include("benchmarkEandFrc.jl")
# include("perfsiteE.jl")


# =========== TEMP =======
