

module NRLTB

using JuLIP
using JuLIP.Potentials
using JuLIP.ASE

using TightBinding: TBModel, SKHamiltonian

import JuLIP.Potentials: evaluate, evaluate_d, grad
import TightBinding: hop!, overlap!, onsite!

export NRLTBModel, NRLHamiltonian

const BOHR = 0.52917721092::Float64  # atomic unit of length 1 Bohr = 0.52917721092 Å
# TODO: insert the BOHR conversion into the hop!, overlap! and onsite! functions

"""
`NRLHamiltonian `: specifies the parameters for the NRLTB hamiltonian
"""
type NRLHamiltonian{NORB} <: SKHamiltonian{NONORTHOGONAL, NORB}

# Norbital: 4 if(s, p) : s, px, py, pz
#           9 if(s, p, d) : s, px, py, pz, dxy, dyz, dzx, dx^2-y^2, d3z^2-r^2
# Nbond: 4 if (s, p) : ssσ, spσ, ppσ, ppπ,
#        10 if (s, p ,d) : ssσ, spσ, ppσ, ppπ, sdσ, pdσ, pdπ, ddσ, ddπ, ddδ
    Norbital::Int
    Nbond::Int

# cutoff parameters
    Rc::Float64
    lc::Float64
# onsite
    λ::Float64
    a::Array{Float64}
    b::Array{Float64}
    c::Array{Float64}
    d::Array{Float64}
# hopping H
    e::Array{Float64}
    f::Array{Float64}
    g::Array{Float64}
    h::Array{Float64}
# hopping M
    p::Array{Float64}
    q::Array{Float64}
    r::Array{Float64}
    s::Array{Float64}
    t::Array{Float64}
end

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
function NRLTBModel(; elem = C_sp, beta=1.0, fixed_eF=true, eF = 0.0,
		    nkpoints = (0, 0, 0), hfd=1e-6)

    onsite = NRLos(elem)
    hop  = NRLhop(elem)
    overlap = NRLoverlap(elem)
    rcut = elem.Rc

    return TBModel(onsite = onsite,
         		   hop = hop,
                   overlap = overlap,
                   Rcut = rcut,
                   smearing = FermiDiracSmearing(beta),
                   norbitals = elem.Norbital,
                   fixed_eF = fixed_eF,
                   eF = eF,
                   nkpoints = nkpoints,
                   hfd = hfd)
end


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

nrl_hop(H, r, i) = (H.e[i] + (H.f[i] + H.g[i] * r) * r) * exp( - H.h[i]^2 * r) *
                  cutoff_NRL(r, H.Rc, H.lc)

function hop!(H::NRLHamiltonian{NORB}, r, temp)
   for i = 1:H.Nbond
      temp[i] = h_hop(H, r, i)
   end
   return temp
end


# ================= OVERLAP INTEGRALS  =====================

nrl_olap(H, r, i) = (H.p[i] + (H.q[i] + (H.r[i] + H.s[i] * r) * r) * r) *
                  exp(-H.t[i]^2 * r) * cutoff_NRL(r, H.Rc, H.lc)

# off-site overlap block
function overlap!(H::NRLHamiltonian, r, temp)
   for i = 1:elem.Nbon
        temp[i] = nrl_olap(H, r, i)
    end
    return temp
end

# on-site overlap block
function overlap!(H::NRLHamiltonian, M_nn)
   M_nn[:,:] = eye(H.Norbital)
   return M_nn
end

# ================= IMPLEMENTATION OF ONSITE FUNCTION =====================

# Preliminaries: Pseudo electron density on site l : ρ_l
# r, R : distances and displacements of the neighboring atoms
# elem : NRLParams
# OUTPUT
# ρ    : return the pseudo density on site n = 1, ... , length(atm)
# note that the NRL pseudo density has ignored the self-distance
function pseudoDensity(H::NRLHamiltonian, r)
    eX = exp(-(H.λ^2) * r)
    fX = cutoff_NRL(r, H.Rc, H.lc)
    return dot(eX, fX)
end

# auxiliary functions for computing the onsite terms
nrl_os(H::NRLHamiltonian, ρ, i) =
   H.a[i] + H.b[i] * ρ^(2/3) + H.c[i] * ρ^(4/3) + H.d[i] * ρ^2

function onsite!(H::NRLHamiltonian, r, _, H_nn)
   ρ = pseudoDensity(H, r)
   fill!(H_nn, 0.0)
   for i = 1:H.Norbital
      H_nn[i, i] = nrl_os(H, ρ)
   end
   return H_nn
end


end
