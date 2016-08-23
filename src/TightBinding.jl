
module TightBinding

using JuLIP
using JuLIP.Potentials

abstract AbstractTBModel <: AbstractCalculator

# general parts of the module: core tb functionality
include("general.jl")

# perturbation theory module
# include("perturbation.jl")

# implements the "classical" site energy
include("site_energy.jl")

# implement the contour integral variants
# in particular the site energy
include("Contour.jl")

# a tight binding toy model, mostly for testing
include("toymodel.jl")

# the NRL tight binding model
include("NRLTB.jl")

end    # module
