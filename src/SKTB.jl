
module SKTB

using JuLIP, StaticArrays
using JuLIP: @protofun
using JuLIP.Potentials: @pot, SitePotential

import JuLIP: energy, forces, cutoff, virial
import JuLIP.Potentials: evaluate, evaluate_d, site_energy, site_energy_d,
                        partial_energy, partial_energy_d

export hamiltonian, densitymatrix, TBModel, SKTBModel


# ============================================================================


# abstractions
include("types.jl")

# chemical potentials
# define how to go from spectrum (energy-levels) to potential energy
include("potentials.jl")

# BZ integration: MPGrid and GammaPoint, iterators, etc
include("bzintegration.jl")

# Construction of contours for PEXSI
include("FermiContour.jl")

# ============= SLATER KOSTER TYPE MODELS ================

# basics for slater-koster type hamiltonians
include("sk_core.jl")

# assembling hamiltonian and hamiltonian derivatives
include("matrixassembly.jl")

# the TB toy model for quick tests (a slater-koster s-orbital model)
include("toymodel.jl")

# the NRLTB model
include("NRLTB.jl")

# The Kwon model - a simple orthogonal TB model for Silicon
include("kwon.jl")

# generic code, such as computing spectral decoposition, energy, forces
include("calculators.jl")

# pole expansion (contour integration) based calculator for TBModel
include("pexsi.jl")

# TODO: perturbation theory module
# include("perturbation.jl")

end    # module
