# SOME NOTES TO REVISIT
# ----------------------
#  * Let's try to get away without definint cutoff for the TBModel ?!?!


export TBModel,
      isorthogonal,
      TBHamiltonian,
      SmearingFunction



# =======================  Abstract Hamiltonian Types =========================

const ORTHOGONAL = :orth
const NONORTHOGONAL = :nonorth


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
type NullHamiltonian{ISORTH} <: TBHamiltonian{ISORTH}
end
NullHamiltonian() = NullHamiltonian{:orth}()

isorthogonal{ISORTH}(::TBHamiltonian{ISORTH}) = ISORTH == ORTHOGONAL


@protofun evaluate(::AbstractAtoms, ::TBHamiltonian, ::AbstractVector)

evaluate(at::AbstractAtoms, H::TBHamiltonian) = evaluate(at, H, JVecF([0.0,0.0,0.0]))


# =======================  Smearing Functions =============================

abstract SmearingFunction

"""
auxiliary smearing function that doesn't do anything but lets us construct
and empty TBModel.
"""
type NullSmearing <: SmearingFunction end



# ===================  TightBinding Calculator =====================


"""
`TBModel`: basic non-self consistent tight binding calculator.
"""
type TBModel{ISORTH} <: AbstractCalculator
   # hamiltonian
   H::TBHamiltonian{ISORTH}
   # additional MM potential (typically but not necessarily pair)
   Vrep::SitePotential
   # smearing function / fermi temperature model
   smearing::SmearingFunction
   # k-point sampling              TODO: should this really be a tuple?
   #    0 = open boundary
   #    1 = Gamma point
   nkpoints::Tuple{Int, Int, Int}

   # -------------- a few internals ------------------
   hfd::Float64           # step used for finite-difference approximations
end

typealias TightBindingModel TBModel

TBModel() = TBModel(NullHamiltonian(), ZeroSitePotential(), NullSmearing(), (0,0,0), 0.0)

isorthogonal(::TBModel{:orth}) = true
isorthogonal(::TBModel{:nonorth}) = false



"""
`hamiltonian`: computes the hamiltonitan and overlap matrix for a tight
binding model.

#### Parameters:

* `atm::AbstractAtoms`
* `tbm::TBModel`
* `k = [0.;0.;0.]` : k-point at which the hamiltonian is evaluated

### Output: H, M

* `H` : hamiltonian in suitable format (typically CSC)
* `M` : overlap matrix in suitable format (typically CSC or I is orthogonal)
"""
hamiltonian(tbm::TBModel, at::AbstractAtoms, args...) = evaluate(tbm.H, at, args...)

evaluate(H::TBHamiltonian, at::AbstractAtoms) = evaluate(H, at, (0,0,0))


# ==============================================================================
