
using TightBinding.FermiContour: fermicontour

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


function set_EminEmax!(at::AbstractAtoms, Emin, Emax)
   set_info!(at, :EminEmax, (Emin, Emax))
end

function get_EminEmax(at::AbstractAtoms)
   Emin, Emax = get_info(at, :EminEmax)::Tuple{Float64, Float64}
   return Emin, Emax
end

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
   # TODO: not so clear that Emin, Emax should be updated!!!!
   Emin, Emax = extrema( abs(epsn - eF) )
   set_EminEmax!(at, Emin, Emax)
   return nothing
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
   @assert fixed_eF(tbm.potential)
   # assume that this is a finite-T model, otherwise we need a different kind
   # of contour - but this should be ok to lift!!!!!
   @assert isa(tbm.potential, FiniteTPotential)

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

   # get the Fermi-contour;
   # to be on the safe side, we don't use the Emin parameter at all; this
   # can cause problems when there is an e-val near eF.
   Emin, Emax = get_EminEmax(at)
   w, z = fermicontour(0.0, Emax, beta(tbm.potential), calc.nquad)
   w = -w    # flip weights due to change in fermicontour implementation
   z += get_eF(tbm.potential)
   Ez = energy(tbm.potential, z)

   # collect all the orbital-indices corresponding to the site-indices into a long vector
   Iorb = indexblock(Is, tbm.H)
   Norb = length(Iorb)
   # define the right-hand sides in the linear solver at each quad-point
   rhsM = full(M[:, Iorb])
   rhs = zeros(size(H,1), Norb);
   rhs[Iorb,:] = eye(Norb)

   # allocate
   E = partial_energy(tbm.Vrep, at, Is)
   # precompute the hamiltonian derivatives
   if deriv
      skhg = SparseSKHgrad(tbm.H, at)
      ∇E = partial_energy_d(tbm.Vrep, at, Is)
   else
      ∇E = zerovecs(length(at))
   end

   # integrate over the contour
   for (wi, zi, Ei) in zip(w, z, Ez)
      # Green's function
      LU = lufact(H - zi * M)
      # --------------- assemble energy -----------
      resM = LU \ rhsM
      E += real(wi * Ei * trace(resM[Iorb,:]))
      # --------------- assemble forces -----------
      if deriv
         res = isorthogonal(tbm) ? resM : LU \ rhs
         _pexsi_site_grad!(∇E, tbm.H, skhg, res, resM, rhs, wi*Ei, zi)
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
         f1 -= (wi * res[In[a], t] * res[Im[b], t]) * dH_nm[:,a,b]
         f1 -= (wi * res[In[a], t] * res[In[b], t]) * dH_nn[:,a,b]
      end
      ∇E[m] += real(f1)
      ∇E[n] -= real(f1)
   end
   return ∇E
end




"""
`calibrate!(calc::PEXSI, at; kwargs...)`

Performans some initialisations / calibrations of the PEXSI Calculator.
"""
function calibrate!(calc::PEXSI, at::AbstractAtoms;
         npoles = nothing, tol = nothing,    # train quadrature rule
         at_train = nothing, nkpoints = nothing, eF = nothing, δNel = 0.0 # train eF, Emin, Emax
   )
   _calibrate_poles!(calc, at, npoles, tol)
   _calibrate_EminEmax!(calc, at, at_train, nkpoints, eF, δNel)
   return  nothing
end

function _calibrate_poles!(calc, at, npoles=nothing, tol=nothing)
   if npoles != nothing && tol != nothing
      error("`calibrate!`: provide *either* `npoles` *or* `tol`")
   end
   if npoles != nothing
      set_npoles!(calc, npoles)
   else
      error("calibration of PEXSI using tolerance is has not been implemented yet.")
   end
end

function _calibrate_EminEmax!(calc, at, at_train, nkpoints, eF, δNel)
   if at_train != nothing && eF != nothing
      error("`calibrate!`: provide either `(at_train, nkpoints)` *or* `eF`")
   end
   if eF != nothing
      set_eF!(calc.tbm, eF)
   else
      @assert isa(calc.tbm.potential, FiniteTPotential)
      # TODO: generalise last line: require that the potential is a
      #       fixed_eF potential, then do the same calibration
      if nkpoints == nothing
         nkpoints = (4,4,4)
      end
      bzquad_at = calc.tbm.bzquad
      calc.tbm.bzquad = MPGrid(at_train, nkpoints)
      update!(at_train, calc.tbm)
      set_δNel!(calc.tbm, at_train, δNel)
      # compute and store Emin, Emax >>> TODO: seems this needs to be moved out of `update!`
      update!(calc, at_train)
      # recover Emin, Emax and store in `at`
      Emin, Emax = get_EminEmax(at_train)
      set_EminEmax!(at, Emin, Emax)
      # finally revert to the original bzquad
      calc.tbm.bzquad = bzquad_at
   end
end
