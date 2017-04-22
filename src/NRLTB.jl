

module NRLTB

using ForwardDiff

using JuLIP.Potentials.ZeroSitePotential
using TightBinding: TBModel, SKHamiltonian, NONORTHOGONAL,
                  norbitals, ChemicalPotential
import TightBinding: hop!, overlap!, onsite!, hop, overlap, onsite_grad!
import JuLIP.Potentials: cutoff

export NRLTBModel, NRLHamiltonian

const BOHR = 0.52917721092::Float64  # atomic unit of length 1 Bohr = 0.52917721092 Å
# const BOHR = 1.0::Float64
# TODO: insert the BOHR conversion into the hop!, overlap! and onsite! functions

# Norbital: 4 if(s, p) : s, px, py, pz
#           9 if(s, p, d) : s, px, py, pz, dxy, dyz, dzx, dx^2-y^2, d3z^2-r^2
# Nbond: 4 if (s, p) : ssσ, spσ, ppσ, ppπ,
#        10 if (s, p ,d) : ssσ, spσ, ppσ, ppπ, sdσ, pdσ, pdπ, ddσ, ddπ, ddδ

"""
`NRLHamiltonian `: specifies the parameters for the NRLTB hamiltonian
"""
immutable NRLHamiltonian{NORB} <: SKHamiltonian{NONORTHOGONAL, NORB}
    Norbital::Int64
    Nbond::Int64
# cutoff parameters
    Rc::Float64
    lc::Float64
# onsite
    λ::Float64
    a::Vector{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    d::Vector{Float64}
# hopping H
    e::Vector{Float64}
    f::Vector{Float64}
    g::Vector{Float64}
    h::Vector{Float64}
# hopping M
    p::Vector{Float64}
    q::Vector{Float64}
    r::Vector{Float64}
    s::Vector{Float64}
    t::Vector{Float64}
end

cutoff(H::NRLHamiltonian) = H.Rc

nbonds(H::NRLHamiltonian) = H.Nbond

# contains information for Si, C, Al
# TODO: switch to data files
include("NRLTB_data.jl")



"""
`NRLTBModel`: constructs the NRL tight binding model.

### Parameters

* elem : NRLParams (default at Carbon atom with s&p orbitals)
* beta = 1.0 : electronic temperature
* fixed_eF = true : if true, then the chemical potential is fixed (default at 0.0)
* eF = 0.0 : chemical potential (if fixed)
* nkpoints : number of k-points at each direction (only (0,0,Int) has been implemented)
* hfd = 1e-6 : finite difference step for computing hessians
"""
NRLTBModel(species, fs::ChemicalPotential;
           orbitals=default_orbitals(species), bzquad=GammaPoint(), hfd=1e-6) =
   TBModel(NRLParams(species, orbitals=orbitals),
           ZeroSitePotential(), fs, bzquad, hfd)


# ================= NRL CUTOFF ===========
# (Modified??) cutoff function in the NRL ansatz for Hamiltonian matrix elements
#    r  : variable
#    Rc : cutoff radius
#    lc : cutoff weight
#    Mc : we changed NRL's 5.0 to 10.0, but I changed it back to 5.0 (CO)
# NB: this cutoff is CRAP - this cut-off is not even differentiable; we should
#     at least force-shift it?!?!?
# TODO: talk to Noam about this.

cutoff_NRL(r, Rc, lc, Mc = 5.0) =
            (1.0 ./ (1.0 + exp( (r-Rc) / lc + Mc ))) .* (r .<= Rc)


# ================= HOPPING INTEGRALS =====================

nrl_hop(H::NRLHamiltonian, r, i) = (H.e[i] + (H.f[i] + H.g[i] * r) * r) * exp( - H.h[i]^2 * r)

hop(H::NRLHamiltonian, r, i) = nrl_hop(H, r/BOHR, i) * cutoff_NRL(r/BOHR, H.Rc, H.lc)

# function hop!(H::NRLHamiltonian, r::Number, temp)
#    r /= BOHR
#    fcut = cutoff_NRL(r, H.Rc, H.lc)
#    for i = 1:nbonds(H)
#       temp[i] = hop(H, r, i) * fcut
#    end
#    return temp
# end


# ================= OVERLAP INTEGRALS  =====================

nrl_olap(H, r, i) = (H.p[i] + (H.q[i] + (H.r[i] + H.s[i] * r) * r) * r) * exp(-H.t[i]^2 * r)

overlap(H::NRLHamiltonian, r::Real, i::Integer) =
      nrl_olap(H, r/BOHR, i) * cutoff_NRL(r/BOHR, H.Rc, H.lc)

# off-site overlap block
# function overlap!(H::NRLHamiltonian, r::Number, temp)
#    r /= BOHR
#    fcut = cutoff_NRL(r, H.Rc, H.lc)
#    for i = 1:nbonds(H)
#         temp[i] = nrl_olap(H, r, i) * fcut
#     end
#     return temp
# end

# on-site overlap block
function overlap!(H::NRLHamiltonian, M_nn)
   fill!(M_nn, 0.0)
   for i = 1:norbitals(H); M_nn[i,i] = 1.0; end
   return M_nn
end

# we don't need a derivative of overlap! (onsite) since it is constant

# ================= IMPLEMENTATION OF ONSITE FUNCTION =====================

# Preliminaries: Pseudo electron density on site l : ρ_l
# r, R : distances and displacements of the neighboring atoms
# elem : NRLParams
# OUTPUT
# ρ    : return the pseudo density on site n = 1, ... , length(atm)
# note that the NRL pseudo density has ignored the self-distance
pseudoDensity(H::NRLHamiltonian, r::AbstractVector) =
   sum( exp(- H.λ^2 * r) .* cutoff_NRL(r, H.Rc, H.lc) )

# auxiliary functions for computing the onsite terms
nrl_os(H::NRLHamiltonian, ρ, i) =
   H.a[i] + H.b[i] * ρ^(2/3) + H.c[i] * ρ^(4/3) + H.d[i] * ρ^2

nrl_os_d(H::NRLHamiltonian, ρ, i) =
   H.b[i] * (2/3) * ρ^(-1/3) + H.c[i] * (4/3) * ρ^(1/3) + H.d[i] * 2 * ρ

function onsite!(H::NRLHamiltonian, r, _, H_nn)
   ρ = pseudoDensity(H, r / BOHR)
   fill!(H_nn, 0.0)
   for i = 1:norbitals(H)
      H_nn[i,i] = nrl_os(H, ρ, i)
   end
   return H_nn
end

function onsite_grad!(H::NRLHamiltonian, r, R, dH_nn)
   ρ = pseudoDensity(H, r / BOHR)
   ∇ρ = ForwardDiff.gradient( r_ -> pseudoDensity(H, r_ / BOHR), r )
   fill!(dH_nn, 0.0)
   for i = 1:norbitals(H), a = 1:3, j = 1:length(r)
      dH_nn[a,i,i,j] = nrl_os_d(H, ρ, i) * ∇ρ[j] * R[j][a] / r[j]
   end
   return dH_nn
end


end
