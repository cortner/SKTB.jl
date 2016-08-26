

"""
This module implements evaluation of TB quantities based on contour integration
instead of spectral decompostions. This module is still missing a lot of
functionality and is therefore still experimental.

Parts of this is based on [PEXSI](https://math.berkeley.edu/~linlin/pexsi/index.html),
but we deviate in various ways. For example, we don't use selected inversion,
but rather (in the future) want to move towards an iterative solver instead.

In fact, the current implementation uses naive direct solvers.

### TODO:
[ ] automatically determine Emin, Emax
[ ] need a 0T contour
[ ] in general: allow different energies, e.g. including entropy
"""
module Contour

using JuLIP
using JuLIP: cutoff
using TightBinding: TBModel, monkhorstpackgrid, hamiltonian, FermiDiracSmearing,
                     update!, band_structure_all, indexblock,
                     evaluate, evaluate_d!, grad!
using FermiContour

export site_energy


type ContourCalculator{P_os, P_hop, P_ol, P_p}
   tbm::TBModel{P_os, P_hop, P_ol, P_p}
   nquad::Int
   Emin::Float64
   Emax::Float64
end

ContourCalculator(tbm, nquad) = ContourCalculator(tbm, nquad, 0.0, 0.0)


"""
uses spectral decomposition to compute Emin, Emax, eF
for the configuration `at` and stores it in `calc`
"""
function calibrate!(calc::ContourCalculator, at::AbstractAtoms,
                     beta::Float64; nkpoints=(4,4,4) )
   tbm = calc.tbm
   tbm.smearing = FermiDiracSmearing(beta)
   tbm.fixed_eF = false
   tbm.eF = 0.0
   tbm.nkpoints, nkpoints_old = nkpoints, tbm.nkpoints
   # this computes the spectrum and fermi-level
   update!(at, tbm)
   @assert tbm.eF == tbm.smearing.eF
   tbm.fixed_eF = true
   tbm.nkpoints = nkpoints_old
   # get the spectrum and compute Emin, Emax
   _, epsn = band_structure_all(at, tbm)
   calc.Emin, calc.Emax = extrema( abs(epsn - tbm.eF) )
   return calc
end


"""
uses spectral decomposition to compute Emin, Emax, eF
for the configuration `at` and stores it in `calc`
"""
function calibrate2!(calc::ContourCalculator, at::AbstractAtoms,
                     beta::Float64; nkpoints=(4,4,4) )
   tbm = calc.tbm
   tbm.smearing = FermiDiracSmearing(beta)
   tbm.fixed_eF = false
   tbm.eF = 0.0
   tbm.nkpoints, nkpoints_old = nkpoints, tbm.nkpoints
   # this computes the spectrum and fermi-level
   H, M = ham(calc.tbm, at)
   e = eigvals(full(H))
   tbm.eF = 0.5 * sum(extrema(e))
   tbm.smearing.eF = tbm.eF
   tbm.fixed_eF = true
   calc.Emin = 0.0
   calc.Emax = maximum( abs(e - tbm.eF) )
   return calc
end



function ham(tbm::TBModel, at::AbstractAtoms)
   I = Int[]
   J = Int[]
   Z = Float64[]
   for (n, neigs, r, R, _) in sites(at, cutoff(tbm))
      In = indexblock(n, tbm)
      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, tbm)
         Hnm = evaluate(tbm.hop, r[i_n], R[i_n])
         for a = 1:tbm.norbitals, b = 1:tbm.norbitals
            push!(I, In[a])
            push!(J, Im[b])
            push!(Z, Hnm[a,b])
         end
      end
   end
   N = length(at) * tbm.norbitals
   return sparse(I, J, Z, N, N), speye(N)
end




# TODO: at the moment we just have a single loop to compute
#       energy and forces; consider instead to have forces separately
#       but store precomputed information (the residuals)

function site_energy(calc::ContourCalculator, at::AbstractAtoms,
                     n0::Integer; deriv=false)
   tbm = calc.tbm

   # ----------- some temporary things to check simplifying assumptions
   # assume that the fermi-level is fixed
   @assert tbm.fixed_eF
   # assume that the smearing function is FermiDiracSmearing
   @assert isa(tbm.smearing, FermiDiracSmearing)
   # assume that we have only one k-point
   # TODO: we will need BZ  integration in at least one coordinate direction
   # K, w = monkhorstpackgrid(at, tbm)
   # @assert length(K) == 1

   # ------------------ main part of the assembly stars here
   # get the hamiltonian for the Gamma point
   H, M = hamiltonian(tbm, at) 
   H = full(H); M = full(M)

   # get the Fermi-contour
   w, z = fermicontour(calc.Emin, calc.Emax, tbm.smearing.beta, tbm.eF, calc.nquad)
   # compute site energy
   # define the right-hand side in the linear solver at each quad-point
   rhsM = M[:, n0]
   rhs = zeros(size(H,1)); rhs[n0] = 1.0

   Esite = 0.0
   Esite_d = zerovecs(length(at))

   for (wi, zi) in zip(w, z)
      LU = lufact(H - zi * M)

      # --------------- assemble energy -----------
      res = LU \ rhs
      resM = LU \ rhsM
      Esite += 2.0 * real(wi * zi * resM[n0])

      # --------------- assemble forces -----------
      if deriv
         Esite_d += site_grad_inner(tbm, at, res, resM, 2.0*wi*zi)
      end
   end
   # --------------------------------------------

   return Esite, Esite_d
end


function site_grad_inner(tbm, at, res, resM, wi)

   dH_nm = zeros(3, tbm.norbitals, tbm.norbitals)
   vdH_nm = dH_nm |> vecs   # no x no matrix  of JVecF

   # allocate force vector
   frc = zerovecs(length(at))

   for (n, neigs, r, R, _) in sites(at, cutoff(tbm))
      In = indexblock(n, tbm)
      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, tbm)
         grad!(tbm.hop, r[i_n], R[i_n], dH_nm)
         f1 = JVec(0.0)
         for a = 1:tbm.norbitals, b = 1:tbm.norbitals
            f1 += - (wi * res[In[a]] * resM[Im[b]]) * vdH_nm[a,b]
         end
         frc[n] -= real(f1)
         frc[m] += real(f1)
      end
   end
   return frc
end   # site_force_inner

end
