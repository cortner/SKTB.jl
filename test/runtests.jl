using TightBinding
using JuLIP, JuLIP.ASE
using JuLIP.Testing
using Base.Test

TESTDEPTH = 1

println("============================================")
println("    TightBinding Tests  ")
println("============================================")

# =========== Main tests =================
include("testtoymodel.jl")
include("testnrltb.jl")
include("testsiteE.jl")

# ============= Compare with Atoms.jl and QUIP implementations
# include("compareatoms.jl")
# include("comparequip.jl")


# include("benchmarkEandFrc.jl")
# include("perfsiteE.jl")
