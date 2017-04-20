
module TightBinding

using JuLIP, StaticArrays
using JuLIP: @protofun
using JuLIP.Potentials: @pot, SitePotential

import JuLIP: energy, forces, cutoff
import JuLIP.Potentials: evaluate, evaluate_d, site_energy

# using FixedSizeArrays
export hamiltonian, densitymatrix, TBModel, TightBindingModel



# ============================================================================


# abstractions
include("types.jl")

# define how to go from eigenvalues (energy-levels) to potential energy
include("smearing.jl")

# BZ integration: basic prototype;
include("bzintegration.jl")

# ============= SLATER KOSTER TYPE MODELS ================

# basics for slater-koster type hamiltonians
include("sk_core.jl")
include("matrixassembly.jl")

# the TB toy model for quick tests (a slater-koster s-orbital model)
include("toymodel.jl")

# the NRLTB model
include("NRLTB.jl")

# The Kwon model - a simple orthogonal TB model for Silicon
# include("kwon.jl")

# generic code, such as computing spectral decoposition,
# energy, forces (given the hamiltonian and hamiltonian derivatives)
# TODO: rename to become eigcalculators or standardcalc
include("calculators.jl")

# implement the contour integral variants of the TB model
# include("contour.jl")


# TODO: perturbation theory module
# include("perturbation.jl")

end    # module
