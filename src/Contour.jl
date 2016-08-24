

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
using TightBinding: TBModel, monkhorstpackgrid, hamiltonian, FermiDiracSmearing,
                     update!, band_structure_all, indexblock
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


"create a canonical basis vector"
en(n::Int, N::Int) = full( sparsevec([n], [1.0], N) )


function site_energy(calc::ContourCalculator, at::AbstractAtoms, n0::Integer)
   tbm = calc.tbm

   # assume that the fermi-level is fixed
   @assert tbm.fixed_eF
   # assume that the smearing function is FermiDiracSmearing
   @assert isa(tbm.smearing, FermiDiracSmearing)
   # assume that we have only one k-point
   # TODO: we will need BZ  integration in at least one coordinate direction
   K, w = monkhorstpackgrid(at, tbm)
   @assert length(K) == 1

   # --------------------------------------------
   # get the hamiltonian for the Gamma point
   H, M = hamiltonian(tbm, at)
   H = full(H); M = full(M)
   # get the Fermi-contour
   w, z = fermicontour(calc.Emin, calc.Emax, tbm.smearing.beta, tbm.eF, calc.nquad)
   # compute site energy
   # define the right-hand side in the linear solver at each quad-point
   In0 = indexblock(n0, tbm) |> Vector
   rhs = full(M[:, In0])
   Esite = 0.0
   for (wi, zi) in zip(w, z)
      res = (H - zi * M) \ rhs
      # TODO: why is there a 2.0 here? It wasn't needed in the initial tests !!!
      Esite += 2.0 * real(wi * zi * trace(res[In0, :]))
   end
   # --------------------------------------------

   return Esite
end


end






# # contour integral version
# function siteE_contour(tb::TBSystem,
#                         n0::Int,   # site index
#                         Emin::Float64, Emax::Float64,  # spectrum bounds
#                         nquad::Int)     # number of quadrature points
#     H = hamiltonian(tb)
#     w, z = fermicontour(Emin, Emax, tb.β, tb.μ, nquad)
#     en = full_en(n0, size(H,1))
#     Esite = 0.0
#     residuals = Vector{Complex128}[]
#     for (wi, zi) in zip(w, z)
#         res = (H - zi * speye(H)) \ en
#         push!(residuals, res)
#         Esite += real(wi * zi * res[n0])
#     end
#     return Esite, Esite_data(n0, w, z, residuals, H)
# end
#
# "extract the indices and values for a single column"
# col(A::SparseMatrixCSC{Float64,Int}, i) = A.rowval[A.colptr[i]:A.colptr[i+1]-1], A.nzval[A.colptr[i]:A.colptr[i+1]-1]
#
# function siteE_d_contour(tb::TBSystem, dat::Esite_data)
#     dH = hamiltonian_d(tb)::SparseMatrixCSC{Float64,Int}
#     N = size(dH, 1)
#     const y = zeros(2)
#     const dEs = zeros(size(tb.Y))::Matrix{Float64}
#     for (wi, zi, r) in zip(dat.w, dat.z, dat.residuals) # sum over contour points
#         for i = 1:N                     # sum over sites on which to compute the force
#             jj, hh = col(dH, i)         # get the i-th column of dH, which contains D_{ji} = D_{ij}
#             for (j, h) in zip(jj, hh)   # loop through non-zero column entries
#                 # make sure this is fast (in practise we should use FixedSizeArrays)
#                 y[1] = tb.Y[1,i] - tb.Y[1,j]
#                 y[2] = tb.Y[2,i] - tb.Y[2,j]
#                 # wrap around
#                 for a = 1:2; y[a] = mod(y[a] + tb.R, 2*tb.R) - tb.R; end
#                 nrm = sqrt(y[1]*y[1]+y[2]*y[2])
#                 a = (2.0 * real(wi * zi * r[i] * r[j]) * dH[i,j]/nrm)
#                 dEs[1, i] -= a * y[1]
#                 dEs[2, i] -= a * y[2]
#             end
#         end
#     end
#     return dEs
# end
#
# Base.println(tb::TBSystem) = println("R = ", tb.R, "; # sites = ", size(tb.Y,2), "; # bonds = ", length(tb.B[1]));
#
