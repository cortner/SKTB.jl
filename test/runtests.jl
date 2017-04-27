using TightBinding
using JuLIP, JuLIP.ASE
using JuLIP.Testing
using Base.Test

COMPAREATOMS = true    # if Atoms.jl is installed
COMPAREQUIP = true
TESTDEPTH = 1

println("============================================")
println("    TightBinding Tests  ")
println("============================================")

# =========== Main tests =================
# include("testtoymodel.jl")
# include("testnrltb.jl")
include("testsiteE.jl")

# ============= Compare with Atoms.jl and QUIP implementations
if COMPAREATOMS; include("compareatoms.jl"); end
if COMPAREQUIP; include("comparequip.jl"); end




# include("benchmarkEandFrc.jl")
# include("perfsiteE.jl")
