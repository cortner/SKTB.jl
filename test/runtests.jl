using TightBinding
using JuLIP
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
include("testcontour.jl")
include("testsiteE.jl")
include("testkwon.jl")
include("testdual.jl")
# include("test0T.jl")

# ============= Compare with Atoms.jl and QUIP implementations
if COMPAREATOMS; include("compareatoms.jl"); end
if COMPAREQUIP; include("comparequip.jl"); end

# ============= Performance benchmarks
# (uncomment these only for performance tests)
# include("benchmarkEandFrc.jl")
# include("perfsiteE.jl")


# =========== TEMP =======
