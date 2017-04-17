
module TightBinding

using JuLIP
using JuLIP: @protofun
using JuLIP.Potentials: @pot, SitePotential

import JuLIP: energy, forces, cutoff
import JuLIP.Potentials: evaluate, evaluate_d

# using FixedSizeArrays
export hamiltonian, densitymatrix




# ============================================================================


# abstractions
include("types.jl")

# define how to go from eigenvalues (energy-levels) to potential energy
include("smearing.jl")

# BZ integration: basic prototype; TODO: implement more efficient BZ grids!
include("bzintegration.jl")

# generic code, such as computing spectral decoposition,
# energy, forces (given the hamiltonian and hamiltonian derivatives)
include("calculators.jl")

# implement the contour integral variants of the TB model
# include("contour.jl")

# ============= SLATER KOSTER TYPE MODELS ================

# basics for slater-koster type hamiltonians
include("slaterkoster.jl")

# the TB toy model for quick tests (a slater-koster s-orbital model)
include("toymodel.jl")

# the NRLTB model
include("NRLTB.jl")

# The Kwon model - a simple orthogonal TB model for Silicon
# include("kwon.jl")






# perturbation theory module
# include("perturbation.jl")

# implements the "classical" site energy
# include("site_energy.jl")

end    # module
