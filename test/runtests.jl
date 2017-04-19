using TightBinding
using JuLIP, JuLIP.ASE
using JuLIP.Testing
using Base.Test

println("============================================")
println("    TightBinding Tests  ")
println("============================================")


# include("testtoymodel.jl")
# include("testnrltb.jl")
# include("testsiteE.jl")
# include("perfsiteE.jl")
# include("comparequip.jl")

# include("compareatoms.jl")


include("benchmarkEandFrc.jl")
