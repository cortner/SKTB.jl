

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
* [ ] automatically determine Emin, Emax
* [ ] 0T contour
"""
type PEXSI{TBM <: TBModel} <: AbstractTBModel
   tbm::TBM
   nquad::Int           # number of quadrature points
   Idom::Vector{Int}    # subset of atoms on which to compute the energy
end


PEXSI(tbm, nquad) = PEXSI(tbm, nquad, Int[])
PEXSI(tbm, nquad, Idom) = PEXSI(tbm, nquad, Idom, 0.0, 0.0)

function set_domain!(calc::PEXSI, Idom::Vector{Int})
   calc.Idom = Idom
   return calc
end

energy(calc::PEXSI, at::AbstractAtoms) =
         partial_energy(calc, at, calc.Idom, false)[1]

forces(calc::PEXSI, at::AbstractAtoms) =
         - partial_energy(calc, at, calc.Idom, true)[2]

site_energy(calc::PEXSI, at::AbstractAtoms, n0::Integer) =
         partial_energy(calc, at, [n0], false)[1]

site_energy_d(calc::PEXSI, at::AbstractAtoms, n0::Integer) =
         partial_energy(calc, at, [n0], true)[2]


"""
`update!(calc::PEXSI, at::AbstractAtoms; δNel = nothing, Nel = nothing)`

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
   set_info!(at, (Emin, Emax), :EminEmax)    # TODO: this is not so clear that it shouldn't be transient!!
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
function partial_energy{TI <: Integer}(
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
   @assert length(calc.tbm.bzquad) == 1

   # ------------------ main part of the assembly starts here
   # get the hamiltonian for the Gamma point
   # this should return either full or sparse depending on the size of the system 
   H, M = hamiltonian(tbm, at)

   # get the Fermi-contour
   w, z = fermicontour(calc.Emin, calc.Emax, tbm.potential.beta, tbm.eF, calc.nquad)

   # collect all the orbital-indices corresponding to the site-indices
   # into a long vector
   Iorb = indexblock(Is, tbm)
   Norb = length(Iorb)
   # define the right-hand sides in the linear solver at each quad-point
   rhsM = M[:, Iorb]
   rhs = zeros(size(H,1), Norb);
   rhs[Iorb,:] = eye(Norb)

   E = 0.0
   ∇E = zerovecs(length(at))

   # integrate over the contour
   for (wi, zi) in zip(w, z)
      # compute the Green's function
      LU = lufact(H - zi * M)

      # --------------- assemble energy -----------
      resM = LU \ rhsM
      # TODO: this 2.0 is to account for an error in the basic TB implementation
      #       where we forgot to account for double-occupancy???
      E += 2.0 * real(wi * zi * trace(resM[Iorb,:]))

      # --------------- assemble forces -----------
      # TODO: the call to site_force_inner will very likely dominate this;
      #       since we are recomputing H_{,n} and H_{,m} many times here
      #       (for each quadrature point on the contour)
      #       it will probably be better to first precompute all residuals
      #       `res`, store them, and then start a new loop over the contour
      #       this is to be tested. For 1_000 atoms, 4 orbitals per atom
      #       and 20 quadrature points, this would amount to ca. 4GB of data
      if deriv
         res = LU \ rhs   # this should cost a fraction of the LU-factorisation
         ∇E += site_grad_inner(tbm, at, res, resM, rhs, 2.0*wi*zi, zi)
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


function site_grad_inner(tbm, at, res, resM, e0, wi, zi)

   # @assert size(res) == size(resM) == size(e0)

   # count the maximum number of neighbours
   nlist = neighbourlist(at, cutoff(tbm))
   # this is a long loop, but it costs nothing compared to the for-loop below
   maxneigs = maximum( length(s[2]) for s in sites(nlist) )

   # pre-allocate dH, dM arrays
   dH_nn = zeros(3, tbm.norbitals, tbm.norbitals, maxneigs)
   dH_nm = zeros(3, tbm.norbitals, tbm.norbitals)
   dM_nm = zeros(3, tbm.norbitals, tbm.norbitals)
   # creates references to these arrays; when dH_nn etc get new data
   # written into them, then vdH_nn etc are automatically updated.
   vdH_nn = vecs(dH_nn)::Array{JVecF, 3}   # no x no x maxneigs array with each entry a JVecF
   vdH_nm = vecs(dH_nm)::Matrix{JVecF}     # no x no matrix  of JVecF
   vdM_nm = vecs(dM_nm)::Matrix{JVecF}     # no x no matrix  of JVecF

   # allocate force vector
   frc = zerovecs(length(at))

   for (n, neigs, r, R, _) in sites(at, cutoff(tbm))
      In = indexblock(n, tbm)
      evaluate_d!(tbm.onsite, r, R, dH_nn)    # 2100 (performance)
      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, tbm)
         grad!(tbm.hop, r[i_n], R[i_n], dH_nm)    #  2600
         grad!(tbm.overlap, r[i_n], R[i_n], dM_nm)   # 2800
         f1 = JVec(0.0im,0.0im,0.0im)
         for t = 1:size(res,2), a = 1:tbm.norbitals, b = 1:tbm.norbitals   # 2500
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
