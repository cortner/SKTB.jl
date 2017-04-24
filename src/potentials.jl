
import Calculus

export ZeroTemperature,
      ZeroTemperatureGrand,
      MerminFreeEnergy,
      GrandPotential,
      FermiDiracSmearing

# a smearing function can be
#  [1] 0T or finite T
#  [2] fixed eF or variable eF

# TODO: consider combining ZeroTemperature with MerminFreeEnergy and
#       ZeroTemperatureGrand with GrandPotential
#
# TODO: ChemicalPotential is not a great term, what would be a better one?


# ChemicalPotentials define:
# * occupancy   n(ϵ)
# * energy   f(ϵ)
# * grad     f'(ϵ)
#     NB: grad is defined via ForwardDiff, but for an efficient implemention
#         this can of course be overloaded

grad(f::ChemicalPotential, epsn::Real) = ForwardDiff.derivative(s->energy(f, s), epsn)
grad(f::ChemicalPotential, epsn::AbstractVector) = [grad(f, s) for s in epsn]

occupancy_d(f::ChemicalPotential, epsn::Real) = ForwardDiff.derivative(s -> occupancy(f, s), epsn)
occupancy_d(f::ChemicalPotential, epsn::Real, μ::Real) =
   ForwardDiff.derivative(s -> occupancy(f, s, μ), epsn)
occupancy_d(f::ChemicalPotential, epsn::AbstractVector, args...) =
   [occupancy_d(f, s, args...) for s in epsn]

beta(f::FiniteTPotential) = f.beta
beta(f::ZeroTPotential) = Inf    # or should it be `nothing`???

# ================= Zero-Temperature Models  ============================


@pot type ZeroTemperatureGrand <: ChemicalPotential
   eF::Float64
end
"""
0T model for the electrons in the grand potential
(with fixed Fermi-level eF, but variable number of particle Nel)
"""
ZeroTemperatureGrand

update!(::AbstractAtoms, ::ZeroTemperatureGrand) = nothing


# TODO: continue here; need an eF for this one as well
#       probably write a collect_epsn function to compute and sort all e-vals
#       then compute eF from those.

@pot type ZeroTemperature <: ChemicalPotential
   Nel::Float64
end
"""
0T model for the electrons in the canonical potentials
(fixed number of particles Nel, variable fermi-level eF)
"""
ZeroTemperature

update!(::AbstractAtoms, ::ZeroTemperature) = nothing

function set_Nel!(f::ZeroTemperature, Nel::Integer)
   f.Nel = Nel
   return f
end

function set_Nel!(tbm::AbstractTBModel, Nel::Integer)
   set_Nel!(tbm.potential, Nel)
   return tbm
end



# ================= Fermi-Dirac Distribution  ========================

# symbolic differentiation of the Fermi-dirac distribution
_fd0_ = :(1.0 / ( 1.0 + exp((epsn - eF) * beta) ))
_fd1_ = Calculus.differentiate(_fd0_, :epsn)
_fd2_ = Calculus.differentiate(_fd1_, :epsn)
_fd3_ = Calculus.differentiate(_fd2_, :epsn)

eval( :( fermidirac(epsn, eF, beta) = $_fd0_ ) )
eval( :( fermidirac1(epsn, eF, beta) = $_fd1_ ) )
eval( :( fermidirac2(epsn, eF, beta) = $_fd2_ ) )
eval( :( fermidirac3(epsn, eF, beta) = $_fd3_ ) )



# ================= Canonical Ensemble (Mermin) ============================

@pot type MerminFreeEnergy <: ChemicalPotential
   Nel::Float64
   beta::Float64
   eF::Float64
end

"""
Mermin Free energy is given by
`e ↦  2 e f(e - μ) + 2 β⁻¹ entropy(f(e-μ))`

where μ is determined from  `2 ∑_s f(e_s - μ) = Nel.`
"""
MerminFreeEnergy

_en0_ = :( f * log(f) - (1-f) * log(1-f) )
_en1_ = Calculus.differentiate(_en0_, :f)
_en2_ = Calculus.differentiate(_en1_, :f)
_en3_ = Calculus.differentiate(_en2_, :f)

eval( :( entropy(f) = $_en0_ ) )
eval( :( entropy1(f) = $_en1_ ) )
eval( :( entropy2(f) = $_en2_ ) )
eval( :( entropy3(f) = $_en3_ ) )


function update!(at::AbstractAtoms, f::MerminFreeEnergy)
   error("`update!` for `MerminFreeEnergy` has not been implemented yet.")
   # need to solve the nonlinear system that ensures
   # ∑_k ∑_s f(ϵ_sk) = Nel/2    (or whatever)
end


# ================= Grand Potential ============================


@pot type GrandPotential <: ChemicalPotential
   beta::Float64
   eF::Float64
end

"""
Finite temperature grand-potential

e ↦ 2/β log(1 - f(e))

where f is the Fermi-dirac function.
"""
GrandPotential

_gr0_ = :( 2.0/beta * log(1 - $_fd0_) )
_gr1_ = Calculus.differentiate(_gr0_, :epsn)
_gr2_ = Calculus.differentiate(_gr1_, :epsn)
_gr3_ = Calculus.differentiate(_gr2_, :epsn)



# ================= The Old Smearing Function ============================
# should still implement it, then deprecate and remove once
# we have fixed the Atoms.jl - TightBinding.jl equivalence

@pot type FermiDiracSmearing  <: FiniteTPotential
    beta::Float64
    eF::Float64
    Ne::Float64
    fixed_eF::Bool
end

get_eF(f::FermiDiracSmearing) = f.eF
get_Ne(f::FermiDiracSmearing) = f.Ne

"""`FermiDiracSmearing`:

f(e) = ( 1 + exp( beta (e - eF) ) )^{-1}
"""
FermiDiracSmearing

FermiDiracSmearing(beta; eF=0.0, Ne = 0.0, fixed_eF = true) = FermiDiracSmearing(beta, eF, Ne, fixed_eF)

occupancy(fd::FermiDiracSmearing, epsn::Number) = fermidirac(epsn, fd.eF, fd.beta)
occupancy(fd::FermiDiracSmearing, epsn::Number, eF) = fermidirac(epsn, eF, fd.beta)
occupancy(fd::FermiDiracSmearing, epsn::AbstractVector, args...) =
      [occupancy(fd, es, args...) for es in epsn]

energy(fd::FermiDiracSmearing, epsn::Number) = fermidirac(epsn, fd.eF, fd.beta) * epsn
energy(fd::FermiDiracSmearing, epsn::AbstractVector) = [energy(fd, es) for es in epsn]


function set_eF!(fd::FermiDiracSmearing, eF)
   fd.eF = eF
end

function update!(at::AbstractAtoms, f::FermiDiracSmearing, tbm::TBModel)
   if !f.fixed_eF
      f.eF = eF_solver(at, f, tbm)
   end
   return nothing
end


occupancy(at::AbstractAtoms, f::ChemicalPotential, tbm, μ = get_eF(f)) =
   sum( w * occupancy(f, epsn, μ) for (w, _k, epsn, _ψ) in BZiter(tbm, at) )

occupancy_d(at::AbstractAtoms, f::ChemicalPotential, tbm, μ = get_eF(f)) =
   sum( w * occupancy_d(f, epsn, μ) for (w, _k, epsn, _ψ) in BZiter(tbm, at) )


# TODO: make this robust by
#  * adding bounds on available Ne
#  * adding a bisection stage
#
function eF_solver(at, f, tbm, Ne = get_Ne(f))
   update!(at, tbm)
   # guess-timate an initial eF (this assumes Fermi-Dirac like behaviour)
   nf = ceil(Int, Ne)
   μ = 0.0
   for (w, k) in tbm.bzquad
      epsn_k = get_k_array(at, :epsn, k)
      μ += w * (epsn_k[nf-1] + epsn_k[nf]) / 2
   end

   # Newton iteration
   err = 1.0
   itctr = 0
   while abs(err) > 1e-8
      Ni = occupancy(at, f, tbm, μ)
      gi = occupancy_d(at, f, tbm, μ)
      err = Ne - Ni
      μ = μ - err / gi
      itctr += 1
      if itctr > 20
         error("eF_solver Newton iteration failed.")
      end
   end

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
