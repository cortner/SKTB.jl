
module Kwon

using Parameters
using TightBinding: SKHamiltonian, hop, TBModel, NullPotential, ORTHOGONAL
using JuLIP.Potentials: fcut, fcut_d, SplineCutoff, PairPotential, ZeroPairPotential, EAM

import JuLIP: cutoff
import TightBinding: hop, onsite!, onsite_grad!

"""
`KwonHamiltonian <: SKHamiltonian{ORTHOGONAL, 4}`

Hamiltonian for an orthogonal sp TB model of Si developed by Kwon et al [1].

This implementation deviates  from [1] in how the cut-off is applied:
instead of "patching" a cubic spline between r1 and rcut, we simply multiply
with a quintic spline on the interval [0.5 (rcut + r0), rcut].

[1] I. Kwon, R. Biswas, C. Z. Wang, K. M. Ho and C. M. Soukoulis.
Transferable tight-binding models for silicon.
Phys Rev B 49 (11), 1994.
"""
@with_kw immutable KwonHamiltonian <: SKHamiltonian{ORTHOGONAL, 4}
   r0::Float64 = 2.360352   # Å
   Es::Float64 = -5.25       # eV
   Ep::Float64 = 1.2        # eV
   E0::Float64 = 8.7393204  # eV   8.7393204  is the original value; but it is just a constant
   # ------------------------------- Electronic Parameters
   # α   1     2    3     4
   #    ssσ   spσ  ppσ   ppπ
   hr0::NTuple{4, Float64} = (-2.038, 1.745, 2.75, -1.075)
   nc::NTuple{4, Float64} = (9.5, 8.5, 7.5, 7.5)
   rc::NTuple{4, Float64} = (3.4, 3.55, 3.7, 3.7)
   # ------------------------------- 2-body parameters
   m::Float64 = 6.8755
   mc::Float64 = 13.017
   dc::Float64 = 3.66995     # Å
   C::NTuple{4, Float64} = (2.1604385, -0.1384393, 5.8398423e-3, -8.0263577e-5)
   # --------------------------------
   r1::Float64 = 3.260176     # Å  (start of cut-off) >>> different r1 from Kwon paper
   rcut::Float64 = 4.16       # Å  (end of cut-off)
end

cutoff(H::KwonHamiltonian) = H.rcut

kwon_hop(H::KwonHamiltonian, r, α) = ( H.hr0[α] * (H.r0 / r)^2 *
               exp( - 2 * (r/H.rc[α])^H.nc[α] + 2 * (H.r0/H.rc[α])^H.nc[α] ) )

hop(H::KwonHamiltonian, r, α) = kwon_hop(H, r, α) * fcut(r, H.r1, H.rcut)

function onsite!(H::KwonHamiltonian, _r, _R, H_nn)
   fill!(H_nn, 0.0)
   H_nn[1,1] = H.Es
   H_nn[2,2] = H_nn[3,3] = H_nn[4,4] = H.Ep
   return H_nn
end

onsite_grad!(H::KwonHamiltonian, _r, _R, dH_nn) = fill!(dH_nn, 0.0)

# ============ Repulsive potential

# embedding function C1 x + C2 x^2 + C3 x^3 + C3 x^4
KwonEmbedding(H::KwonHamiltonian) = PairPotential(
   :( $(H.E0) + r * ($(H.C[1]) + r * ($(H.C[2]) + r * ($(H.C[3]) + r * $(H.C[4])))) ) )

# electron density: (r0/r)^m * exp( - m * (r/dc)^mc + m * (r0/dc)^mc )
KwonElDensity(H::KwonHamiltonian) = ( PairPotential(
   :( ($(H.r0)/r)^($(H.m)) * exp( - $(H.m) * (r/$(H.dc))^($(H.mc))
                                  + $(H.m) * ($(H.r0/H.dc))^($(H.mc)) ) )
   ) * SplineCutoff(H.r1, H.rcut) )

KwonEAM(H::KwonHamiltonian) = EAM(ZeroPairPotential(), KwonElDensity(H), KwonEmbedding(H))


function KwonTBModel(; potential = NullPotential(), bzquad = NullBZQ())
   H = KwonHamiltonian()
   return TBModel(H, KwonEAM(H), potential, bzquad, 1e-6)
end

end
