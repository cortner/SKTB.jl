
import Calculus

export ZeroTemperature,
      ZeroTemperatureGrand,
      MerminFreeEnergy,
      GrandPotential

# a smearing function can be
#  [1] 0T or finite T
#  [2] fixed eF or variable eF

# TODO: consider combining ZeroTemperature with MerminFreeEnergy and
#       ZeroTemperatureGrand with GrandPotential
#
# TODO: SmearingFunction is not a great term, what would be a better one?


# SmearingFunctions define:
# * occupancy   n(ϵ)
# * energy   f(ϵ)
# * grad     f'(ϵ)
#     NB: grad is defined via ForwardDiff, but for an efficient implemention
#         this can of course be overloaded

grad(f::SmearingFunction, epsn::Real) = ForwardDiff.derivative(s->energy(f, s), epsn)
grad(f::SmearingFunction, epsn::AbstractVector) = [grad(f, s) for s in epsn]


# ================= Zero-Temperature Models  ============================


@pot type ZeroTemperatureGrand <: SmearingFunction
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

@pot type ZeroTemperature <: SmearingFunction
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
   set_Nel!(tbm.smearing, Nel)
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

@pot type MerminFreeEnergy <: SmearingFunction
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


@pot type GrandPotential <: SmearingFunction
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

@pot type FermiDiracSmearing  <: SmearingFunction
    beta::Float64
    eF::Float64
    fixed_eF::Bool
end

"""`FermiDiracSmearing`:

f(e) = ( 1 + exp( beta (e - eF) ) )^{-1}
"""
FermiDiracSmearing

FermiDiracSmearing(beta; eF=0.0) = FermiDiracSmearing(beta, eF)

occupancy(fd::FermiDiracSmearing, epsn::Number) = fermidirac(epsn, fd.eF, fd.beta)
occupancy(fd::FermiDiracSmearing, epsn::AbstractVector) = [occupancy(fd, es) for es in epsn]

energy(fd::FermiDiracSmearing, epsn::Number) = fermidirac(epsn, fd.eF, fd.beta) * epsn
energy(fd::FermiDiracSmearing, epsn::AbstractVector) = [energy(fd, es) for es in epsn]




# # Boilerplate to work with the FermiDiracSmearing type
# evaluate(fd::FermiDiracSmearing, epsn) = fermidirac(epsn, fd.eF, fd.beta, epsn)
# evaluate_d(fd::FermiDiracSmearing, epsn) = fermidirac1(fd.eF, fd.beta, epsn)
# # Boilerplate to work with the FermiDiracSmearing type
# evaluate(fd::FermiDiracSmearing, epsn, eF) = fermi_dirac(eF, fd.beta, epsn)
# evaluate_d(fd::FermiDiracSmearing, epsn, eF) = fermi_dirac_d(eF, fd.beta, epsn)

function set_eF!(fd::FermiDiracSmearing, eF)
   fd.eF = eF
end

function update!(at::AbstractAtoms, f::FermiDiracSmearing)
   if !f.fixed_eF
      error("update! for FermiDiracSmearing has not been implemented yet")
   end
   return nothing
end



# TODO: The code below is the eF-solver that needs to be plugged into the
#       canonical ensemble smearing functions

# """
# `update_eF!(tbm::TBModel)`: recompute the correct
# fermi-level; using the precomputed data in `tbm.arrays`
# """
# function update_eF!(atm::AbstractAtoms, tbm::TBModel)
#    if tbm.fixed_eF
#       set_eF!(tbm.smearing, tbm.eF)
#       return
#    end
#    # the following algorithm works for Fermi-Dirac, not general Smearing
#    K, weight = monkhorstpackgrid(atm, tbm)
#    Ne = tbm.norbitals * length(atm)
#    nf = round(Int, ceil(Ne/2))
#    # update_eig!(atm, tbm)
#    # set an initial eF
#    μ = 0.0
#    for n = 1:length(K)
#       k = K[n]
#       epsn_k = get_k_array(tbm, :epsn, k)
#       μ += weight[n] * (epsn_k[nf] + epsn_k[nf+1]) /2
#    end
#    # iteration by Newton algorithm
#    err = 1.0
#    while abs(err) > 1.0e-8
#       Ni = 0.0
#       gi = 0.0
#       for n = 1:length(K)
#          k = K[n]
#          epsn_k = get_k_array(tbm, :epsn, k)
#          Ni += weight[n] * sum_kbn( tbm.smearing(epsn_k, μ) )
#          gi += weight[n] * sum_kbn( @D tbm.smearing(epsn_k, μ) )
#       end
#       err = Ne - Ni
#       #println("\n err=");  print(err)
#       μ = μ - err / gi
#    end
#    tbm.eF = μ
#    set_eF!(tbm.smearing, tbm.eF)
# end
