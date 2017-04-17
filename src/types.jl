# SOME NOTES TO REVISIT
# ----------------------
#  * Let's try to get away without definint cutoff for the TBModel ?!?!


export AbstractTBModel,
      TBModel,
      isorthogonal,
      TBHamiltonian,
      SmearingFunction,
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
returns total number of electron dofs
"""
@protofun ndofs(::TBHamiltonian, ::AbstractAtoms)

@protofun evaluate(::TBHamiltonian, ::AbstractAtoms, ::AbstractVector)

evaluate(H::TBHamiltonian, at::AbstractAtoms) = evaluate(H, at, JVecF([0.0,0.0,0.0]))


# =======================  Smearing Functions =============================

abstract SmearingFunction

"""
auxiliary smearing function that doesn't do anything but lets us construct
and empty TBModel.
"""
type NullSmearing <: SmearingFunction end


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
type TBModel{ISORTH} <: AbstractTBModel
   # hamiltonian
   H::TBHamiltonian{ISORTH}
   # additional MM potential (typically but not necessarily pair)
   Vrep::SitePotential
   # smearing function / fermi temperature model   TODO: smearing is not a good name for this?
   smearing::SmearingFunction
   # k-point sampling
   bzquad::BZQuadratureRule
   # -------------- internals ------------------
   # step used for finite-difference approximations
   hfd::Float64
end

typealias TightBindingModel TBModel

TBModel() = TBModel(NullHamiltonian(), ZeroSitePotential(),
                    NullSmearing(), GammaPoint(), 0.0)

isorthogonal(::TBModel{:orth}) = true
isorthogonal(::TBModel{:nonorth}) = false



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
hamiltonian(tbm::AbstractTBModel, at::AbstractAtoms, args...) = evaluate(tbm.H, at, args...)



# ==============================================================================
