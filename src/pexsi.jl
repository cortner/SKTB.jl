
using FermiContour

"""
`PEXSI <: AbstractTBModel`:

This type implements evaluation of TB quantities based on contour integration
/ pole expansion instead of spectral decompostion.
Parts of this is based on [PEXSI](https://math.berkeley.edu/~linlin/pexsi/index.html),
but we deviate in various ways. For example, we don't use selected inversion
at the moment. The current implementation uses naive direct solvers.

This calculator is still missing some
functionality and is therefore experimental.

### TODO:
* [ ] an automatic calibration mechanism that determines a good
      discretisation (nquad) as well as E0, E1.
* [ ] 0T contour
"""
type PEXSI{TBM <: TBModel} <: AbstractTBModel
   tbm::TBM
   nquad::Int           # number of quadrature points
   Idom::Vector{Int}    # subset of atoms on which to compute the energy
end


PEXSI(tbm, nquad) = PEXSI(tbm, nquad, Int[])
PEXSI(tbm, nquad, Idom) = PEXSI(tbm, nquad, Idom)

function set_domain!(calc::PEXSI, Idom::Vector{Int})
   calc.Idom = Idom
   return calc
end

"""
`set_npoles!(calc::PEXSI, n)`: set the number of poles (quadrature points)
in the PEXSI scheme
"""
function set_npoles!(calc::PEXSI, n)
   calc.nquad = n
end

energy(calc::PEXSI, at::AbstractAtoms) =
         pexsi_partial_energy(calc, at, calc.Idom, false)[1]

forces(calc::PEXSI, at::AbstractAtoms) =
         - pexsi_partial_energy(calc, at, calc.Idom, true)[2]

site_energy(calc::PEXSI, at::AbstractAtoms, n0::Integer) =
         pexsi_partial_energy(calc, at, [n0], false)[1]

site_energy_d(calc::PEXSI, at::AbstractAtoms, n0::Integer) =
         pexsi_partial_energy(calc, at, [n0], true)[2]


"""
`update!(calc::PEXSI, at::AbstractAtoms)`

uses spectral decomposition to pre-compute some parameters needed for the PEXSI
scheme; in particular  Emin, Emax, eF for the configuration `at` and stores it
in `calc`. This is normally done in a preprocessing step before the actual
computation.
"""
function update!(calc::PEXSI, at::AbstractAtoms)
   update!(at, calc.tbm)
   # get the spectrum and compute Emin, Emax
   epsn = spectrum(calc.tbm, at)
   eF = get_eF(calc.tbm)
   Emin, Emax = extrema( abs(epsn - eF) )
   set_info!(at, :EminEmax, (Emin, Emax))    # TODO: this is not so clear that it shouldn't be transient!!
   return nothing
end
# TODO: not so clear that Emin, Emax should be updated!!!!

import Base.\
function \(A::Base.SparseArrays.UMFPACK.UmfpackLU, B::Matrix{Float64})
   out = zeros(B)
   for n = 1:size(B, 2)
      out[:, n] = A \ B[:, n]
   end
   return out
end


"""
partial_energy(calc::PEXSI, at, Is, deriv=false)

Instead of the total energy of a QM system this computes the energy stored in
a sub-domain defined by `Is`.

* `calc`: a `PEXSI`, defining a tight-binding model
* `at`: an atoms object
* `Is`: a list (`AbstractVector`) of indices specifying the subdomain
* `deriv`: whether or not to compute derivatives as well
"""
function pexsi_partial_energy{TI <: Integer}(
                     calc::PEXSI, at::AbstractAtoms,
                     Is::AbstractVector{TI}, deriv=false)
   tbm = calc.tbm
   EminEmax = get_info(at, :EminEmax)

   # ----------- some temporary things to check simplifying assumptions
   # assume that the fermi-level is fixed
   @assert tbm.potential.fixed_eF
   # assume that the smearing function is FermiDiracSmearing
   # but this should be ok to lift!!!!!
   @assert isa(tbm.potential, FermiDiracSmearing)

   # assume that we have only one k-point
   # TODO: eventually (when implementing dislocations) we will need BZ
   #       integration in at least one coordinate direction
   # K, w = monkhorstpackgrid(at, tbm)
   @assert isa(tbm.bzquad, GammaPoint)

   # ------------------ main part of the assembly starts here
   # get the hamiltonian for the Gamma point
   # this should return either full or sparse depending on the size of the system
   H, M = hamiltonian(tbm, at)
   # for now convert `I` into `speye`
   if isa(M, UniformScaling)
      M = speye(ndofs(tbm.H, at))
   end

   # get the Fermi-contour
   Emin, Emax = get_info(at, :EminEmax)::Tuple{Float64, Float64}
   w, z = fermicontour(Emin, Emax, beta(tbm.potential), get_eF(tbm.potential), calc.nquad)

   # collect all the orbital-indices corresponding to the site-indices
   # into a long vector
   Iorb = indexblock(Is, tbm.H)
   Norb = length(Iorb)
   # define the right-hand sides in the linear solver at each quad-point
   rhsM = full(M[:, Iorb])
   rhs = zeros(size(H,1), Norb);
   rhs[Iorb,:] = eye(Norb)

   E = 0.0
   ∇E = zerovecs(length(at))

   # prepare for force assembly
   if deriv
      skhg = SparseSKHgrad(tbm.H, at)
   end

   # integrate over the contour
   for (wi, zi) in zip(w, z)
      # compute the Green's function
      LU = lufact(H - zi * M)

      # --------------- assemble energy -----------
      resM = LU \ rhsM
      E += real(wi * zi * trace(resM[Iorb,:]))

      # --------------- assemble forces -----------
      if deriv
         if isorthogonal(tbm)
            res = resM
         else
            res = LU \ rhs
         end
         ∇E += _pexsi_site_grad(tbm, at, tbm.H, skhg, res, resM, rhs, wi*zi, zi)
         # ∇E += _site_grad_inner(tbm, at, res, resM, rhs, wi*zi, zi)
         # # >>>>>>>>> START DEBUG >>>>>>>>
         # # (keep this code for performance testing)
         # Profile.clear()
         # @profile  Esite_d += site_grad_inner(tbm, at, res, resM, rhs, 2.0*wi*zi, zi)
         # Profile.print()
         # quit()
         # # <<<<<<<<< END DEBUG <<<<<<<<<
      end
   end
   # --------------------------------------------
   return E, ∇E
end


# ~~~~~~~~~~~~~~ CONTINUE HERE ~~~~~~~~~~~~~~~

# import Base.getindex


function _pexsi_site_grad{ISORTH, NORB}(tbm, at, H::SKHamiltonian{ISORTH, NORB},
                                        skhg, res, resM, e0, wi, zi)

   # allocate force vector
   ∇E = zerovecs(length(at))

   if isorthogonal(tbm)
      dM_ij = @SArray zeros(3, norbitals(tbm), norbitals(tbm))
   end
   f1 = zeros(Complex128, 3)

   for n = 1:length(skhg.i)
      i, j, dH_ij, dH_ii, S = skhg.i[n], skhg.j[n], skhg.dH[n], skhg.dOS[n], skhg.Rcell[n]
      if !isorthogonal(tbm); dM_ij = skhg.dM[n]; end
      Ii, Ij = indexblock(i, H), indexblock(j, H)
      fill!(f1, 0.0) #JVec(0.0im,0.0im,0.0im)
      for t = 1:size(res,2), a = 1:NORB, b = 1:NORB
         for c = 1:3
            f1[c] += - (wi * res[Ii[a], t] * resM[Ij[b], t]) *
                               ( dH_ij[c,a,b] - zi * dM_ij[c,a,b] )
            f1[c] += - (wi * res[Ii[a], t] * resM[Ii[b], t]) * dH_ii[c,a,b]
            f1[c] += (wi * res[Ii[a], t] * e0[Ij[b], t]) * dM_ij[c,a,b]
         end
      end
      ∇E[j] += real(f1)
      ∇E[i] -= real(f1)
   end

   return ∇E
end




function _site_grad_inner(tbm, at, res, resM, e0, wi, zi)

   # count the maximum number of neighbours
   nlist = neighbourlist(at, cutoff(tbm.H))
   # this is a long loop, but it costs nothing compared to the for-loop below
   maxneigs = maximum( length(s[2]) for s in sites(nlist) )

   # pre-allocate dH, dM arrays
   norb = norbitals(tbm.H)
   dH_nn = zeros(3, norb, norb, maxneigs)
   dH_nm = zeros(3, norb, norb)
   dM_nm = zeros(3, norb, norb)
   # creates references to these arrays; when dH_nn etc get new data
   # written into them, then vdH_nn etc are automatically updated.
   vdH_nn = vecs(dH_nn)::Array{JVecF, 3}   # no x no x maxneigs array with each entry a JVecF
   vdH_nm = vecs(dH_nm)::Matrix{JVecF}     # no x no matrix  of JVecF
   vdM_nm = vecs(dM_nm)::Matrix{JVecF}     # no x no matrix  of JVecF

   const bonds = zeros(nbonds(H))
   const dbonds = zeros(nbonds(H))

   # allocate force vector
   frc = zerovecs(length(at))

   for (n, neigs, r, R, _) in sites(at, cutoff(tbm.H))
      In = indexblock(n, tbm.H)
      onsite_grad!(tbm.H, r, R, dH_nn)    # 2100 (performance)
      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, tbm.H)

         # hop_d!(tbm.hop, r[i_n], R[i_n], dH_nm)    #  2600
         # grad!(tbm.overlap, r[i_n], R[i_n], dM_nm)   # 2800

         hop_d!(H, r[i_n], bonds, dbonds)
         sk_d!(H, r[i_n], R[i_n], bonds, dbonds, dH_nm)
         if !ISORTH
            overlap_d!(H, r[i_n], bonds, dbonds)
            sk_d!(H, r[i_n], R[i_n], bonds, dbonds, dM_nm)
         end

         f1 = JVec(0.0im,0.0im,0.0im)
         for t = 1:size(res,2), a = 1:norb, b = 1:norb   # 2500
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
