
using Parameters
import Roots
import Calculus

export ZeroTemperature,
      ZeroTemperatureGrand,
      MerminFreeEnergy,
      GrandPotential,
      FermiDiracSmearing

# a smearing function can be
#  [1] 0T or finite T
#  [2] fixed eF or variable eF
# TODO: ZeroTemperature models, MerminFreeEnergy


# ChemicalPotentials define:
# * occupancy   n(ϵ)
# * energy   f(ϵ)
# * grad     f'(ϵ)
#     NB: grad is defined via ForwardDiff, but for an efficient implemention
#         this can of course be overloaded

# AD derivative of potential and occupancy
grad(f::FiniteTPotential, epsn::Real) =
      ForwardDiff.derivative(s -> energy(f, s), epsn)
occupancy_d(f::FiniteTPotential, epsn::Real, args...) =
      ForwardDiff.derivative(s -> occupancy(f, s, args...), epsn)

# vectorized versions
energy(f::ChemicalPotential, epsn::AbstractVector, args...) =
      [energy(f, es, args...) for es in epsn]
occupancy(f::ChemicalPotential, epsn::AbstractVector, args...) =
      [occupancy(f, es, args...) for es in epsn]
grad(f::ChemicalPotential, epsn::AbstractVector, args...) =
      [grad(f, s, args...) for s in epsn]
occupancy_d(f::FiniteTPotential, epsn::AbstractVector, args...) =
      [occupancy_d(f, s, args...) for s in epsn]

# default implementations for extracting beta and eF
beta(f::FiniteTPotential) = f.beta
get_eF(f::FiniteTPotential) = f.eF

function set_eF!(fd::ChemicalPotential, eF)
   fd.eF = eF
end

function set_beta!(fd::FiniteTPotential, beta)
   fd.beta = beta
end

# total occupancy calculation

occupancy(at::AbstractAtoms, f::ChemicalPotential, tbm, μ = get_eF(f)) =
   sum( w * occupancy(f, epsn, μ) for (w, _k, epsn, _ψ) in BZiter(tbm, at) )

occupancy_d(at::AbstractAtoms, f::ChemicalPotential, tbm, μ = get_eF(f)) =
   sum( w * occupancy_d(f, epsn, μ) for (w, _k, epsn, _ψ) in BZiter(tbm, at) )



# ================= Fermi-Dirac Distribution  ========================

# symbolic differentiation of the Fermi-dirac distribution
_fd0_ = :(1.0 / ( 1.0 + exp((epsn - eF) * beta) ))
_fd1_ = Calculus.differentiate(_fd0_, :epsn)
_fd2_ = Calculus.differentiate(_fd1_, :epsn)
_fd3_ = Calculus.differentiate(_fd2_, :epsn)

eval( :( fermidirac(epsn::Number, eF, beta) = $_fd0_ ) )
eval( :( fermidirac1(epsn, eF, beta) = $_fd1_ ) )
eval( :( fermidirac2(epsn, eF, beta) = $_fd2_ ) )
eval( :( fermidirac3(epsn, eF, beta) = $_fd3_ ) )

fermidirac(epsn::AbstractVector, eF, beta) = [fermidirac(e, eF, beta) for e in epsn]



# ================= Grand Potential ============================


mutable struct GrandPotential <: FiniteTPotential
   beta::Float64
   eF::Float64
end

@pot GrandPotential

"""
Finite temperature grand-potential, e ↦ 2/β log(1 - f(e)),
where f is the Fermi-dirac function.
"""
GrandPotential

GrandPotential(; β=30.0, beta=β, eF = 0.0) = GrandPotential(beta, eF)

occupancy(f::GrandPotential, epsn::Number, eF) = fermidirac(epsn, eF, f.beta)
occupancy(f::GrandPotential, epsn::Number) = occupancy(f, epsn, f.eF)

# _logfd(z) = real(z) < 0 ? z - log(1+exp(z)) : log( exp(z) / (1+exp(z)) )
_logfd(z) = real(z) < 0 ? z - log1p(exp(z)) : -log1p(exp(-z));
# _logfd(z) = log( exp(z) / (1+exp(z)) )
# _logfd(z) = z - log(1+exp(z))
grand(epsn, eF, beta) = 2.0/beta * _logfd(beta*(epsn-eF))

energy(f::GrandPotential, epsn::Number) = grand(epsn, f.eF, f.beta)

update!(at::AbstractAtoms, f::GrandPotential, tbm::TBModel) = nothing

set_Nel!(f::GrandPotential, tbm, at, Nel) = set_eF!(f, eF_solver(at, f, tbm, Nel))

# get_Ne(f::GrandPotential) = occupancy(at::AbstractAtoms, f::ChemicalPotential, tbm, μ = get_eF(f))

fixed_eF(::GrandPotential) = true


# ================= The Old Smearing Function ============================
# should still implement it, then deprecate and remove once
# we have fixed the Atoms.jl - TightBinding.jl equivalence

mutable struct FermiDiracSmearing  <: FiniteTPotential
    beta::Float64
    eF::Float64
    Ne::Float64
    fixed_eF::Bool
end

@pot FermiDiracSmearing

get_Ne(f::FermiDiracSmearing) = f.Ne
fixed_eF(f::FermiDiracSmearing) = f.fixed_eF


"""`FermiDiracSmearing`:

f(e) = ( 1 + exp( beta (e - eF) ) )^{-1}

Constructor: `FermiDiracSmearing(beta; eF=0.0, Ne = 0.0, fixed_eF = true)`
"""
FermiDiracSmearing

FermiDiracSmearing(beta; eF=0.0, Ne = 0.0, fixed_eF = true) =
      FermiDiracSmearing(beta, eF, Ne, fixed_eF)

occupancy(fd::FermiDiracSmearing, epsn::Number) = fermidirac(epsn, fd.eF, fd.beta)
occupancy(fd::FermiDiracSmearing, epsn::Number, eF) = fermidirac(epsn, eF, fd.beta)
energy(fd::FermiDiracSmearing, epsn::Number) = fermidirac(epsn, fd.eF, fd.beta) * epsn

function update!(at::AbstractAtoms, f::FermiDiracSmearing, tbm::TBModel)
   if !f.fixed_eF
      f.eF = eF_solver(at, f, tbm)
   end
   return nothing
end

"""
`eF_solver(at, f, tbm, Ne = get_Ne(f))`

* `at`: `AbstractAtoms`
* `f`: potential (e.g. `FermiDiracSmearing`, `GrandPotential`, etc)
* `tbm`: `TightBindingModel`
* `Ne` : number of electrons in the system

Given an electron number `Ne` compute the fermi-level (chemical potential)
such that `∑_s f_s = Ne` (where `f_s` is the occupation number)
"""
function eF_solver(at, f, tbm, Ne = get_Ne(f))
   update!(at, tbm)
   # guess-timate an initial eF (this assumes Fermi-Dirac like behaviour)
   nf = ceil(Int, Ne)
   μ = 0.0
   for (w, k) in tbm.bzquad
      epsn_k = get_k_array(at, :epsn, k)
      μ += w * (epsn_k[nf-1] + epsn_k[nf]) / 2
   end
   # call the Roots package
   μ = Roots.fzero( _μ -> Ne - occupancy(at, f, tbm, _μ), μ )
   @assert abs(Ne - occupancy(at, f, tbm, μ)) < 1e-6

   return μ
end



"""
`set_δNel!(tbm::TBModel, at::AbstractAtoms, δNel = 0.0)`

Set the particle number (number of electrons) to `(ndofs(at) + δNel) / 2`.
"""
set_δNel!(tbm::TBModel, at::AbstractAtoms, δNel = 0.0) =
   set_Nel!(tbm, at, (ndofs(tbm.H, at)+δNel)/2)

set_Nel!(tbm::TBModel, at::AbstractAtoms, Nel) =
   set_Nel!(tbm.potential, tbm, at, Nel)


function set_Nel!(f::FermiDiracSmearing, tbm, at, Nel)
   f.Ne = Nel
   f.eF = eF_solver(at, f, tbm, Nel)
   return nothing
end



# ================= Zero-Temperature Models  ============================

function fermilevel(tbm::TBModel, at::AbstractAtoms, Nel)
   e = spectrum(tbm, at) |> sort
   Nel = length(e) ÷ 2
   # Nel = floor(Int, Nel)
   return 0.5 * (e[Nel] + e[Nel+1])
end

abstract type ZeroTPotential <: ChemicalPotential end

"""
`ZeroT`: 0T canonical model (Nel fixed)
"""
@with_kw mutable struct ZeroT <: ZeroTPotential
   Nel::Float64 = 0.0
   eF::Float64 = 0.0
end

"""
`ZeroTGrand`: 0T Grand-canonical model (eF fixed)
"""
@with_kw mutable struct ZeroTGrand <: ZeroTPotential
   Nel::Float64 = 0.0
   eF::Float64 = 0.0
end

beta(f::ZeroTPotential) = 0.0

get_eF(f::ZeroTPotential) = f.eF

get_Nel(f::ZeroTPotential) = f.Nel

occupancy(f::ZeroTPotential, epsn::Number) = occupancy(f, epsn, f.eF)
occupancy(f::ZeroTPotential, epsn::Number, μ) = epsn < μ ? 1.0 : 0.0

energy(f::ZeroTPotential, epsn::Number) = occupancy(f, epsn) * epsn

grad(f::ZeroTPotential, epsn::Number) = occupancy(f, epsn)

function set_Nel!(f::ZeroTPotential, tbm, at, Nel)
   f.Nel = Nel
   f.eF = fermilevel(tbm, at, Nel)
   return nothing
end

function update!(at::AbstractAtoms, f::ZeroT, tbm::TBModel)
   f.eF = fermilevel(tbm, at, f.Nel)
   return nothing
end

function update!(at::AbstractAtoms, f::ZeroTGrand, tbm::TBModel)
   f.Nel = occupancy(at, f, tbm)
   return nothing
end





# # ================= Canonical Ensemble (Mermin) ============================
#
# @pot mutable struct MerminFreeEnergy <: ChemicalPotential
#    Nel::Float64
#    beta::Float64
#    eF::Float64
# end
#
# """
# Mermin Free energy is given by
# `e ↦  2 e f(e - μ) + 2 β⁻¹ entropy(f(e-μ))`
#
# where μ is determined from  `2 ∑_s f(e_s - μ) = Nel.`
# """
# MerminFreeEnergy
#
# _en0_ = :( f * log(f) - (1-f) * log(1-f) )
# _en1_ = Calculus.differentiate(_en0_, :f)
# _en2_ = Calculus.differentiate(_en1_, :f)
# _en3_ = Calculus.differentiate(_en2_, :f)
#
# eval( :( entropy(f) = $_en0_ ) )
# eval( :( entropy1(f) = $_en1_ ) )
# eval( :( entropy2(f) = $_en2_ ) )
# eval( :( entropy3(f) = $_en3_ ) )
#
#
# function update!(at::AbstractAtoms, f::MerminFreeEnergy)
#    error("`update!` for `MerminFreeEnergy` has not been implemented yet.")
#    # need to solve the nonlinear system that ensures
#    # ∑_k ∑_s f(ϵ_sk) = Nel/2    (or whatever)
# end
