# SOME NOTES TO REVISIT
# ----------------------
#  * Let's try to get away without definint cutoff for the TBModel ?!?!

using JuLIP: AbstractAtoms, AbstractCalculator

export AbstractTBModel,
      TBModel,
      isorthogonal,
      TBHamiltonian,
      ChemicalPotential,
      BZQuadratureRule




# =======================  Abstract Hamiltonian Types =========================

const ORTHOGONAL = true
const NONORTHOGONAL = false


"""
abstract Tight Binding Hamiltonian operator; the parameter
`ISORTH` is used to  allow dispatch w.r.t. whether the TB model is orthogonal
   or non-orthogonal
"""
abstract type TBHamiltonian{ISORTH} end

"""
An auxiliary hamiltonian that doesn't do anything but lets us
construct an empty TBModel.
"""
mutable struct NullHamiltonian <: TBHamiltonian{ORTHOGONAL}
end

isorthogonal(::TBHamiltonian{ISORTH}) where {ISORTH} = (ISORTH == ORTHOGONAL)
isorth(H::TBHamiltonian) = isorthogonal(H)

"""
`ndofs(::TBHamiltonian, ::AbstractAtoms)`

returns total number of electron dofs
"""
function ndofs end

evaluate(H::TBHamiltonian, at::AbstractAtoms) = evaluate(H, at, JVecF([0.0,0.0,0.0]))


"""
`SKHamiltonian{ISORTH, NORB}`

abstract supertype for all Slater-Koster type TB hamiltonians. Type parameters
are

* `ISORTH` ∈ {`true`, `false`} : determine whether TB model is orthogonal or non-orthogonal
* `NORB` ∈ {1,4,9} : integer, number of orbitals.
"""
abstract type SKHamiltonian{ISORTH, NORB} <: TBHamiltonian{ISORTH} end

norbitals(::SKHamiltonian{ISORTH, NORB}) where {ISORTH,NORB} = NORB

nbonds(::SKHamiltonian{ISORTH, 1}) where {ISORTH} = 1
nbonds(::SKHamiltonian{ISORTH, 4}) where {ISORTH} = 4
nbonds(::SKHamiltonian{ISORTH, 9}) where {ISORTH} = 10

ndofs(H::SKHamiltonian, at::AbstractAtoms) = norbitals(H) * length(at)


# =======================  Smearing Functions =============================

"""
`ChemicalPotential`: abstract supertype for different chemical potentials.
`SKTB.jl` implements:
# TODO: make list
"""
abstract type ChemicalPotential end

abstract type FiniteTPotential <: ChemicalPotential end
abstract type ZeroTPotential <: ChemicalPotential end

"""
auxiliary smearing function that doesn't do anything but lets us construct
and empty TBModel.
"""
mutable struct NullPotential <: ChemicalPotential end


# ======================= BZ Quadrature supertype =====================
# see `bzintegration.jl` for implementations

"""
`BZQuadratureRule`: abstract BZ quadrature supertype. Quadrature rules
can be applied either useing `w_and_pts` or through iterators, e.g.,
```
for (w, k) in tbm.bzquad
    ...
end
```
"""
abstract type BZQuadratureRule end

mutable struct NullBZQ <: BZQuadratureRule end

# ===================  Standard SKTB Calculator =====================

"""
supertype for all TB model type calcualtors
"""
abstract type AbstractTBModel <: AbstractCalculator end

"""
`TBModel`: basic non-self consistent tight binding calculator.
"""
mutable struct TBModel{HT <: TBHamiltonian, ST <: ChemicalPotential} <: AbstractTBModel
   H::HT                 # hamiltonian
   Vrep::SitePotential   # additional MM potential (typically but not necessarily pair)
   potential::ST         # chemical potential / smearing function
   bzquad::BZQuadratureRule      # k-point sampling
   # -------------- internals ------------------
   hfd::Float64          # step used for finite-difference approximations
end

const SKTBModel = TBModel

TBModel() = TBModel(NullHamiltonian(), ZeroSitePotential(),
                    NullPotential(), NullBZQ(), 0.0)

isorthogonal(tbm::TBModel) = isorthogonal(tbm.H)
isorth(tbm::TBModel) = isorthogonal(tbm)

get_eF(tbm::TBModel) = get_eF(tbm.potential)

norbitals(tbm::TBModel) = norbitals(tbm.H)

"""
`hamiltonian`: computes the hamiltonitan and overlap matrix for a tight
binding model.

#### Parameters:

* `tbm::TBModel`
* `atm::AbstractAtoms`
* `k = [0.;0.;0.]` : k-point at which the hamiltonian is evaluated

### Output: (H, M)

* `H` : hamiltonian in suitable format (typically CSC)
* `M` : overlap matrix in suitable format (typically CSC or I if orthogonal)
"""
hamiltonian(tbm::AbstractTBModel, at::AbstractAtoms, args...) =
   evaluate(tbm.H, at, args...)

# TODO: return full or sparse hamiltonian depending on sparsity: 0.06% seems a good heuristic!


# ==============================================================================
