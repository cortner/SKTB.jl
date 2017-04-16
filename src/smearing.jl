
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

FermiDiracSmearing(beta;eF=0.0) = FermiDiracSmearing(beta, eF)


# Boilerplate to work with the FermiDiracSmearing type
evaluate(fd::FermiDiracSmearing, epsn) = fermi_dirac(fd.eF, fd.beta, epsn)
evaluate_d(fd::FermiDiracSmearing, epsn) = fermi_dirac_d(fd.eF, fd.beta, epsn)
# Boilerplate to work with the FermiDiracSmearing type
evaluate(fd::FermiDiracSmearing, epsn, eF) = fermi_dirac(eF, fd.beta, epsn)
evaluate_d(fd::FermiDiracSmearing, epsn, eF) = fermi_dirac_d(eF, fd.beta, epsn)

function set_eF!(fd::FermiDiracSmearing, eF)
   fd.eF = eF
end

function update!(at::AbstractAtoms, f::FermiDiracSmearing)
   if !fixed_eF
      error("update! for FermiDiracSmearing has not been implemented yet")
   end
   return nothing 
end
