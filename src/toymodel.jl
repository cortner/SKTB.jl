

module ToyModels

#######################################################################
###    The s-orbital Toy model                                      ###
#######################################################################

export ToyTBModel

using JuLIP
using JuLIP.Potentials, JuLIP.ASE
using TightBinding: FermiDiracSmearing, TBModel

import JuLIP.Potentials: evaluate, evaluate_d, @pot


@pot type ToyTBOverlap <: PairPotential
end
# return 0.0 if two parameters are passed (only for off-diagional terms)
evaluate(p::ToyTBOverlap, r, R) = 0.0
evaluate_d(p::ToyTBOverlap, r, R) = 0.0
# return 1.0 for diagonal terms (when r = 0)
evaluate(p::ToyTBOverlap, r) = (r == 0.0 ? 1.0 : error("ToyTBOverlap(r) : r must be 0.0"))


"""
`ToyTBModel`: constructs a simple 1-orbital tight binding model.
It doesn't model anything but can be used for quick tests. The hopping function
is given by

  h(r) = MORSE(r; alpha, r0) * η(r; rcut)

### Parameters

* alpha = 2.0, r0 = 1.0, rcut = 2.7  : Morse potential parameters
* beta = 1.0 : electronic temperature
* fixed_eF = true : if true, then the chemical potential is fixed (default at 0.0)
* eF = 0.0 : chemical potential (if fixed)
* hfd = 1e-6 : finite difference step for computing hessians
"""
function ToyTBModel(;alpha=2.0, r0=1.0, rcut=2.5, beta=1.0, fixed_eF=true,
   eF = 0.0, hfd=1e-6)

   hop = Morse(e0=1.0, A=alpha, r0=r0) * SWCutoff(rcut, 1.0)

   return TBModel(hop = hop,
                  overlap = ToyTBOverlap(),
                  smearing = FermiDiracSmearing(beta),
                  norbitals = 1,
                  Rcut = rcut,
                  fixed_eF = fixed_eF,
                  eF = eF,
                  nkpoints = (0,0,0),
                  hfd=hfd)
end




# ###################### TEST (ENERGY/FORCE) FUNCTIONS OF TOYTB ###############
# function potential_energy_(atm::ASEAtoms, tbm::TBModel)
#     Natm = length(atm)
#     i, j, r = MatSciPy.neighbour_list(atm, "ijd", cutoff(tbm))
#     h = tbm.hop(r)
#     H = sparse(i, j, h, Natm, Natm)
#     epsn, C = sorted_eig(H, speye(Natm))
#     f = tbm.smearing(epsn, tbm.eF)
#     E = r_sum(f .* epsn)
#     return E
# end

# function forces_(atm::ASEAtoms, tbm::TBModel)
#     Natm = length(atm)
#     i, j, r, R = MatSciPy.neighbour_list(atm, "ijdD", cutoff(tbm))

#     # recompute hamiltonian and spectral decomposition
#     h = tbm.hop(r)
#     H = sparse(i, j, h, Natm, Natm)
#     epsn, C = sorted_eig(H, speye(Natm))
#     df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))
#     # compute derivatives of hopping
#     dhop = @D tbm.hop(r)

#     frc = zeros(3, Natm)
#     for a = 1:3
#         # dH = sparse(i, j, dhop .* (R[a,j] - R[a,i])' ./ r, Natm, Natm)
#         dH = sparse(i, j, dhop .* (-R[:,a]) ./ r, Natm, Natm)
#         dH_x_C = dH * C
#         for s = 1:Natm
#             frc[a,:] += 2.0 * df[s] .* C[:,s]' .* dH_x_C[:,s]'
#         end
#     end
#     return frc
# end


end
