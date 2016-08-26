

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
   H, M = hamiltonian(calc.tbm, at)
   e = eigvals(full(H), full(M))
   tbm.eF = 0.5 * sum(extrema(e))
   tbm.smearing.eF = tbm.eF
   tbm.fixed_eF = true
   calc.Emin = 0.0
   calc.Emax = maximum( abs(e - tbm.eF) )
   return calc
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

   # define the right-hand side in the linear solver at each quad-point
   # In0 = indexblock(n0, tbm) |> Vector
   In0 = [1]
   rhsM = M[:, In0]
   rhs = zeros(size(H,1),length(In0)); rhs[In0,:] = eye(length(In0))

   Esite = 0.0
   Esite_d = zerovecs(length(at))

   for (wi, zi) in zip(w, z)
      LU = lufact(H - zi * M)

      # --------------- assemble energy -----------
      resM = LU \ rhsM
      # TODO: why is there a 2.0 here? It wasn't needed in the initial tests !!!
      Esite += 2.0 * real(wi * zi * resM[n0])

      # --------------- assemble forces -----------
      # TODO: the call to site_force_inner will very likely dominate this;
      #       since we are recomputing H_{,n} and H_{,m} many times here
      #       (for each quadrature point on the contour)
      #       it will probably be better to first precompute all residuals
      #       `res`, store them, and then start a new loop over the contour
      #       this is to be tested.
      if deriv
         res = LU \ rhs
         Esite_d += site_grad_inner(tbm, at, res, resM, rhs, 2.0*wi*zi, zi)
      end
   end
   # --------------------------------------------

   return Esite, Esite_d
end

±{T1,T2}(a::Tuple{T1,T2}, b) = (a[1]+b, a[2]-b)

function site_grad_inner(tbm, at, res, resM, e0, wi, zi)

   @assert size(res) == size(resM)
   @assert size(res) == size(e0)

   # count the maximum number of neighbours
   nlist = neighbourlist(at, cutoff(tbm))
   maxneigs = maximum( length(s[2]) for s in sites(nlist) )

   # pre-allocate dH, dM arrays
   dH_nn = zeros(3, tbm.norbitals, tbm.norbitals, maxneigs)
   dH_nm = zeros(3, tbm.norbitals, tbm.norbitals)
   dM_nm = zeros(3, tbm.norbitals, tbm.norbitals)
   # creates references to these arrays; when dH_nn etc get new data
   # written into them, then vdH_nn etc are automatically updated.
   vdH_nn = dH_nn |> vecs   # no x no x maxneigs array with each entry a JVecF
   vdH_nm = dH_nm |> vecs   # no x no matrix  of JVecF
   vdM_nm = dM_nm |> vecs   # no x no matrix  of JVecF

   # allocate force vector
   frc = zerovecs(length(at))

   for (n, neigs, r, R, _) in sites(at, cutoff(tbm))
      In = indexblock(n, tbm)
      evaluate_d!(tbm.onsite, r, R, dH_nn)

      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, tbm)
         grad!(tbm.hop, r[i_n], R[i_n], dH_nm)
         grad!(tbm.overlap, r[i_n], R[i_n], dM_nm)    # DEBUG
         f1 = JVec(0.0)
         for t = 1:1, a = 1:tbm.norbitals, b = 1:tbm.norbitals
            f1 += - (wi * res[In[a], t] * resM[Im[b], t]) *
                               ( vdH_nm[a,b] - zi * vdM_nm[a,b] )
            f1 += - (wi * res[In[a], t] * resM[In[b], t]) * vdH_nn[a,b,i_n]
            f1 += (wi * res[In[a], t] * e0[Im[b], t]) * vdM_nm[a,b]
         end
         frc[m] += real(f1)
         frc[n] -= real(f1)
      end
   end
   return frc
end   # site_force_inner

end
