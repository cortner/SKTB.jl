
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

partial_energy(calc::PEXSI, at::AbstractAtoms, Idom) =
         pexsi_partial_energy(calc, at, Idom, false)[1]

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
   w, z = fermicontour(Emin, Emax, beta(tbm.potential), get_eF(tbm.potential),
                        calc.nquad)

   # collect all the orbital-indices corresponding to the site-indices into a long vector
   Iorb = indexblock(Is, tbm.H)
   Norb = length(Iorb)
   # define the right-hand sides in the linear solver at each quad-point
   rhsM = full(M[:, Iorb])
   rhs = zeros(size(H,1), Norb);
   rhs[Iorb,:] = eye(Norb)

   # allocate
   E = 0.0
   ∇E = zerovecs(length(at))
   # precompute the hamiltonian derivatives
   if deriv
      skhg = SparseSKHgrad(tbm.H, at)
   end

   # integrate over the contour
   for (wi, zi) in zip(w, z)
      # Green's function
      LU = lufact(H - zi * M)
      # --------------- assemble energy -----------
      resM = LU \ rhsM
      E += real(wi * zi * trace(resM[Iorb,:]))
      # --------------- assemble forces -----------
      if deriv
         res = isorthogonal(tbm) ? resM : LU \ rhs
         _pexsi_site_grad!(∇E, tbm.H, skhg, res, resM, rhs, wi*zi, zi)
      end
   end
   return E, ∇E
end



# import Base.getindex
# getindex(a::SArray, ::Colon, i, j) = JVecF(a[1,i,j], a[2,i,j], a[3,i,j])

function _pexsi_site_grad!{NORB}(∇E, H::SKHamiltonian{NONORTHOGONAL,NORB}, skhg,
                                         res, resM, e0, wi, zi)
   for t = 1:length(skhg.i)
      n, m, dH_nm, dH_nn, dM_nm = skhg.i[t], skhg.j[t], skhg.dH[t], skhg.dOS[t], skhg.dM[t]
      In, Im = indexblock(n, H), indexblock(m, H)
      f1 = JVec(0.0im,0.0im,0.0im)
      for t = 1:size(res,2), a = 1:NORB, b = 1:NORB
         f1 += - (wi * res[In[a], t] * resM[Im[b], t]) * ( dH_nm[:,a,b] - zi * dM_nm[:,a,b] )
         f1 += - (wi * res[In[a], t] * resM[In[b], t]) * dH_nn[:,a,b]
         f1 += (wi * res[In[a], t] * e0[Im[b], t]) * dM_nm[:,a,b]
      end
      ∇E[m] += real(f1)
      ∇E[n] -= real(f1)
   end
   return ∇E
end


function _pexsi_site_grad!{NORB}(∇E, H::SKHamiltonian{ORTHOGONAL,NORB}, skhg,
                                 res, resM, e0, wi, zi)
   for t = 1:length(skhg.i)
      n, m, dH_nm, dH_nn = skhg.i[t], skhg.j[t], skhg.dH[t], skhg.dOS[t]
      In = indexblock(n, H)
      Im = indexblock(m, H)
      f1 = JVec(0.0im,0.0im,0.0im)
      for t = 1:size(res,2), a = 1:NORB, b = 1:NORB
         f1 -= (res[In[a], t] * resM[Im[b], t]) * dH_nm[:,a,b]
         f1 -= (res[In[a], t] * resM[In[b], t]) * dH_nn[:,a,b]
      end
      ∇E[m] += wi * real(f1)
      ∇E[n] -= wi * real(f1)
   end
   return ∇E
end
