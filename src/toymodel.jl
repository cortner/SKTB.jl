

#######################################################################
###    The s-orbital Toy model                                      ###
#######################################################################

export ToyTBModel
export ToyHamiltonian

using JuLIP.Potentials: PairPotential, Morse, SWCutoff, ZeroSitePotential

"""
`ToyHamiltonian`: constructs a simple s-orbital tight binding hamiltonian.
The hopping function
is given by any pair potential, but the default is the morse potential

  h(r) = e0 ⋅ e^{- α (r/r0 - 1)} ⋅ fcut(r; rcut)

where fcut is a Stillinger-Weber type cutoff.

See also `ToyTBModel`.

### Constructors:

* `ToyHamiltonian(V)`  where `V <: PairPotential`
* `ToyHamiltonian(; kwargs...)` where admissible kwargs with defaults are
   `alpha = 2.0, r0 = 1.0, rcut = 2.5, e0 = 10.0`.
"""
mutable struct ToyHamiltonian{VT <: PairPotential} <: SKHamiltonian{ORTHOGONAL, 1}
   V::VT
end

ToyHamiltonian(; alpha = 2.0, r0 = 1.0, rcut = 2.5, e0 = 10.0) =
   ToyHamiltonian(Morse(e0=e0, A=alpha, r0=r0) * SWCutoff(rcut, 1.0))

cutoff(H::ToyHamiltonian) = cutoff(H.V)

hop(H::ToyHamiltonian, r::Real, ::Integer) = H.V(r)

onsite!(::ToyHamiltonian, _1, _2, H_nn) = setindex!(H_nn, 0.0, 1)

onsite_grad!(::ToyHamiltonian, _1, _2, dH_nn) = fill!(dH_nn, 0.0)



"""
`ToyTBModel`: constructs a simple 1-orbital SK-type tight binding model,
with Hamiltonian given by `ToyHamiltonian`. It doesn't model anything but can
be used for quick tests, e.g. in conjunction
with the MaterialsScienceTools.TriangularLattice module.

### Keyword Parameters

* alpha = 2.0, r0 = 1.0, rcut = 2.7  : Morse potential parameters
* beta = 1.0 : Fermi temperature
* fixed_eF = true : if true, then the chemical potential is fixed (default at 0.0)
* eF = 0.0 : chemical potential (if fixed)
* hfd = 1e-6 : finite difference step for computing hessians
"""
function ToyTBModel(; beta = 1.0, fixed_eF = true, eF = 0.0,
             hfd = 1e-6, bzquad = GammaPoint(), kwargs...)
   H = ToyHamiltonian(;kwargs...)
   TBModel( H, ZeroSitePotential(),
            FermiDiracSmearing(beta, eF, 0, fixed_eF), bzquad, hfd )
end
