
module TightBinding

using JuLIP
using JuLIP.Potentials
using JuLIP.ASE   # TODO: remove ASE dependence??? 


abstract AbstractTBModel <: AbstractCalculator

# general parts of the module: core tb functionality
include("general.jl")

# perturbation theory module
# include("perturbation.jl")

# implement the contuour integral variants
# include("contour.jl")

# a tight binding toy model, mostly for testing
include("toymodel.jl")

# the NRL tight binding model
# include("NRLTB.jl")


end # module
