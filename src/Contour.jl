

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
                     beta::Float64; nkpoints=(5,5,5) )
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
   K, w = monkhorstpackgrid(at, tbm)
   @assert length(K) == 1

   # ------------------ main part of the assembly stars here
   # get the hamiltonian for the Gamma point
   H, M = hamiltonian(tbm, at)
   H = full(H); M = full(M)
   # get the Fermi-contour
   w, z = fermicontour(calc.Emin, calc.Emax, tbm.smearing.beta, tbm.eF, calc.nquad)
   # compute site energy
   # define the right-hand side in the linear solver at each quad-point
   In0 = indexblock(n0, tbm) |> Vector
   rhsM = full(M[:, In0])
   rhs = zeros(size(H,1), length(In0))
   rhs[In0, :] = eye(length(In0))

   # ****************
   # DEBUG: restrict to first orbital index
   rhs = rhs[:, 1]
   rhsM = rhsM[:, 1]
   In0 = In0[1]
   # ****************

   Esite = 0.0
   Esite_d = zerovecs(length(at))

   for (wi, zi) in zip(w, z)
      LU = lufact(H - zi * M)

      # --------------- assemble energy -----------
      resM = LU \ rhsM
      res = LU \ rhs
      # TODO: why is there a 2.0 here? It wasn't needed in the initial tests !!!
      # Esite += 2.0 * real(wi * zi * trace(resM[In0, :]))
      # DEBUG
      Esite += 2.0 * real(wi * zi * res[In0])

      # --------------- assemble forces -----------
      # TODO: the call to site_force_inner will very likely dominate this;
      #       since we are recomputing H_{,n} and H_{,m} many times here
      #       (for each quadrature point)
      #       it will probably be better to first precompute all residuals
      #       `res`, store them, and then start a new loop over the contour
      #       this is to be tested.
      if deriv
         # res = LU \ rhs
         Esite_d += site_force_inner(tbm, at, res, resM, zi, 2.0*wi*zi)
      end
   end
   # --------------------------------------------

   return Esite, Esite_d
end

# TODO: this is probably both inelegant and inefficient
#       just want correctness for now
_dot_(a, b) = sum( a_*b_ for (a_,b_) in zip(a,b) )
function _dot_{T}(a, M::Matrix{T}, b)
   out = zero(T)
   for i=1:length(a), j = 1:length(b)
      out += (conj(a[j])* b[j]) * M[i,j]
   end
   return out
end


function site_force_inner(tbm, at, res, resM, zi, wi)

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
   X = positions(at)

   for (n, neigs, r, R, _) in sites(nlist)
      In = indexblock(n, tbm) |> Array
      # on-site terms
      # evaluate_d!(tbm.onsite, r, R, dH_nn)
      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, tbm) |> Array
         # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
         grad!(tbm.hop, r[i_n], R[i_n], dH_nm)
         # grad!(tbm.overlap, r[i_n], - R[i_n], dM_nm)

         # >>>>>>>>> START DEBUG >>>>>>>>
         f1 = zero(JVecF)
         for a = 1:length(In), b = 1:length(m)
            f1 += res[In[a]] * res[Im[b]] * vdH_nm[a,b]
         end
         frc[n] += real(wi*f1)
         frc[m] -= real(wi*f1)
         # <<<<<<<<< END DEBUG <<<<<<<<<

         # # TODO:
         # # * move this a-loop into _dot_
         # # * replace copies with views
         # # * WARNING: when we add k-point integration we will need eik multipliers
         # #            in this loop; check that _dot_ handles this correctly
         # for a = 1:tbm.norbitals
         #    # hopping terms
         #    f1 = _dot_(res[In,a], vdH_nm, resM[Im,a])
         #    # f2 = _dot_(res[In,a], vdM_nm, resM[Im,a])
         #    # f3 = _dot_(res[In,a], vdM_nm[:,a])
         #
         #    # on-site term
         #    # f4 = _dot_(res[In,a], vdH_nn[:,:,m], resM[Im,a])
         #
         #    # write contributions into global force vector
         #    #           f3         f2           f1 + f4
         #    #       Rz M_{,n} - (z M_{,n} M - H_{,n} M)
         #    # fall = real( wi * (f3 - zi * f2 + f1 + f4) )
         #    fall = real( wi * f1 )
         #    frc[n] += fall
         #    frc[m] -= fall
         # end
      end
   end
   return frc
end   # site_force_inner

end



# # >>>>>>>>> START DEBUG >>>>>>>>
# S = R[i_n] |> Vector
# Hnm = evaluate(tbm.hop, norm(S), S |> JVec)
# for p = 2:12
#    h = 0.1^p
#    dHh = zeros(dH_nm)
#    for i = 1:3
#       S[i] += h
#       dHh[i,:,:] = (evaluate(tbm.hop, norm(S), S |> JVec) - Hnm) / h
#       S[i] -= h
#    end
#    println(vecnorm(dHh - dH_nm, Inf))
# end
#
# quit()
# # <<<<<<<<< END DEBUG <<<<<<<<<
