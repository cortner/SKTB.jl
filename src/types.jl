# SOME NOTES TO REVISIT
# ----------------------
#  * Let's try to get away without definint cutoff for the TBModel ?!?!


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
abstract TBHamiltonian{ISORTH}

"""
An auxiliary hamiltonian that doesn't do anything but lets us
construct an empty TBModel.
"""
type NullHamiltonian <: TBHamiltonian{ORTHOGONAL}
end

isorthogonal{ISORTH}(::TBHamiltonian{ISORTH}) = ISORTH == ORTHOGONAL
isorth(H::TBHamiltonian) = isorthogonal(H)

"""
`ndofs(::TBHamiltonian, ::AbstractAtoms)`

returns total number of electron dofs
"""
@protofun ndofs(::TBHamiltonian, ::AbstractAtoms)

@protofun evaluate(::TBHamiltonian, ::AbstractAtoms, ::AbstractVector)

evaluate(H::TBHamiltonian, at::AbstractAtoms) = evaluate(H, at, JVecF([0.0,0.0,0.0]))


"""
`SKHamiltonian{ISORTH, NORB}`

abstract supertype for all Slater-Koster type TB hamiltonians. Type parameters
are

* `ISORTH` ∈ {`true`, `false`} : determine whether TB model is orthogonal or non-orthogonal
* `NORB` ∈ {1,4,9} : integer, number of orbitals.
"""
abstract SKHamiltonian{ISORTH, NORB} <: TBHamiltonian{ISORTH}

norbitals{ISORTH,NORB}(::SKHamiltonian{ISORTH, NORB}) = NORB

nbonds{ISORTH}(::SKHamiltonian{ISORTH, 1}) = 1
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 4}) = 4
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 9}) = 10

ndofs(H::SKHamiltonian, at::AbstractAtoms) = norbitals(H) * length(at)


# =======================  Smearing Functions =============================

"""
`ChemicalPotential`: abstract supertype for different chemical potentials.
`TightBinding.jl` implements:
# TODO: make list
"""
abstract ChemicalPotential

abstract FiniteTPotential <: ChemicalPotential
abstract ZeroTPotential <: ChemicalPotential

"""
auxiliary smearing function that doesn't do anything but lets us construct
and empty TBModel.
"""
type NullPotential <: ChemicalPotential end


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
abstract BZQuadratureRule

type NullBZQ <: BZQuadratureRule end

# ===================  Standard TightBinding Calculator =====================

"""
supertype for all TB model type calcualtors
"""
abstract AbstractTBModel <: AbstractCalculator

"""
`TBModel`: basic non-self consistent tight binding calculator.
"""
type TBModel{HT <: TBHamiltonian, ST <: ChemicalPotential} <: AbstractTBModel
   H::HT                 # hamiltonian
   Vrep::SitePotential   # additional MM potential (typically but not necessarily pair)
   potential::ST         # chemical potential / smearing function
   bzquad::BZQuadratureRule      # k-point sampling
   # -------------- internals ------------------
   hfd::Float64          # step used for finite-difference approximations
end

typealias TightBindingModel TBModel

TBModel() = TBModel(NullHamiltonian(), ZeroSitePotential(),
                    NullPotential(), NullBZQ(), 0.0)

isorthogonal(tbm::TBModel) = isorthogonal(tbm.H)
isorth(tbm::TBModel) = isorthogonal(tbm)

get_eF(tbm::TBModel) = get_eF(tbm.potential)

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
