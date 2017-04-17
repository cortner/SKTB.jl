
import JuLIP.Potentials: evaluate, evaluate_d

# slaterkoster.jl
#
# Collect all generic stuff for Slater-Koster-type Tight-binding
# models (which is 99% of non-selfconsistent TB models)
#

abstract SKHamiltonian{ISORTH, NORB} <: TBHamiltonian{ISORTH}

norbitals{ISORTH,NORB}(::SKHamiltonian{ISORTH, NORB}) = NORB

nbonds{ISORTH}(::SKHamiltonian{ISORTH, 1}) = 1
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 4}) = 4
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 9}) = 10

ndofs(H::TBHamiltonian, at::AbstractAtoms) = norbitals(H) * length(at)

############################################################
### indexing for SKHamiltonians

skindexblock(n::Integer, norb::Integer) = Int[(n-1) * norb + j for j = 1:norb]

"""
`indexblock`:
a little auxiliary function to compute indices of Slater Koster orbitals
"""
indexblock(n::Integer, H::SKHamiltonian) = skindexblock(n, norbitals(H))


function skindexblock{T <: Integer}(Iat::AbstractVector{T}, norb::Integer)
   out = T[]
   for n in Iat
      append!(out, indexblock(n, norb))
   end
   return out
end

indexblock{T <: Integer}(Iat::AbstractVector{T}, H::SKHamiltonian) =
   skindexblock(Iat, norbitals(H))





############################################################
### Hamiltonian entries

# the following file contains the real computational core
include("sk_core.jl")

sk!{IO}(H::SKHamiltonian{IO, 1}, U, bonds, out) = bonds[1]

sk!{IO}(H::SKHamiltonian{IO, 4}, U, bonds, out) = sk4!(U, bonds, out)

sk!{IO}(H::SKHamiltonian{IO, 9}, U, bonds, out) = sk9!(U, bonds, out)


############################################################
### Hamiltonian Evaluation
#   most of this could become a *generic* code!
#   it just needs generalising the atom-orbital-dof map.
#   TODO: do this!

function estimate_nnz(H::SKHamiltonian, at::AbstractAtoms)
   nlist = neighbourlist(at, cutoff(H))
   norb = norbitals(H)
   return length(nlist) * norb^2 + length(at) * norb^2
end

function evaluate(H::SKHamiltonian, at::AbstractAtoms, k::AbstractVector)
   nlist = neighbourlist(at, cutoff(H))
   # pre-allocate memory for the triplet format
   nnz_est = estimate_nnz(H, at)
   It = zeros(Int32, nnz_est)
   Jt = zeros(Int32, nnz_est)
   Ht = zeros(Complex{Float64}, nnz_est)
   if isorthogonal(H)
      return assemble!( H, k, It, Jt, Ht, nothing, nlist, positions(at))
   else
      Mt = zeros(Complex{Float64}, nnz_est)
      return assemble!( H, k, It, Jt, Ht, Mt, nlist, positions(at))
   end
   error("nomansland")
end


# TODO: the following method overload are a bit of a hack
#       it would be better to implement broadcasting in FixedSizeArrays

import Base.-
-(A::AbstractVector{JVecF}, a::JVecF) = JVecF[v - a for v in A]

dott(a::JVecF, A::AbstractVector{JVecF}) = JVecF[dot(a, v) for v in A]


# TODO: hide all this sparse matrix crap in a nice type

# append to triplet format: version 1 for H and M (non-orth TB)
function _append!(H::TBHamiltonian{NONORTHOGONAL},
                  It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, norbitals, idx)
   @inbounds for i = 1:norbitals, j = 1:norbitals
      idx += 1
      It[idx] = In[i]
      Jt[idx] = Im[j]
      Ht[idx] = H_nm[i,j] * exp_i_kR
      Mt[idx] = M_nm[i,j] * exp_i_kR
   end
   return idx
end

# append to triplet format: version 2 for H only (orthogonal TB)
function _append!(H::TBHamiltonian{ORTHOGONAL},
                  It, Jt, Ht, _Mt_, In, Im, H_nm, _Mnm_, exp_i_kR, norbitals, idx)
   @inbounds for i = 1:norbitals, j = 1:norbitals
      idx += 1
      It[idx] = In[i]
      Jt[idx] = Im[j]
      Ht[idx] = H_nm[i,j] * exp_i_kR
   end
   return idx
end

# prototypes for the functions needed in `assemble!`
#  TODO: switch to @protofun
function hop! end
function overlap!  end
function onsite! end

# inner Hamiltonian assembly:  non-orthogonal SK tight-binding
#
# exp_i_kR = complex multiplier needed for BZ integration
# note: we could use cell * S instead of R[m] - (X[neigs[m]] - X[n])
#       but this would actually be less efficient, and less clear
#
function assemble!{ISORTH, NORB}(H::SKHamiltonian{ISORTH, NORB},
                                 k, It, Jt, Ht, Mt, nlist, X)

   # TODO: H_nm, M_nm, bonds could all be MArrays
   idx = 0                     # initialise index into triplet format
   H_nm = zeros(NORB, NORB)    # temporary arrays for computing H and M entries
   M_nm = zeros(NORB, NORB)
   bonds = zeros(nbonds(H))     # temporary array for storing the potentials

   # loop through sites
   for (n, neigs, r, R, _) in sites(nlist)
      In = indexblock(n, H)     # index-block for atom index n
      # loop through the neighbours of the current atom (i.e. the bonds)
      for m = 1:length(neigs)
         U = R[m]/r[m]
         # compute hamiltonian block
         sk!(H, U, hop!(H, r[m], bonds), H_nm)
         # compute overlap block, but only if the model is NONORTHOGONAL
         ISORTH || sk!(H, U, overlap!(H, r[m], bonds), M_nm)
         # add new indices into the sparse matrix
         Im = indexblock(neigs[m], H)
         exp_i_kR = exp( im * dot(k, R[m] - (X[neigs[m]] - X[n])) )
         idx = _append!(H, It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, NORB, idx)
      end

      # now compute the on-site blocks;
      # TODO: revisit this (can one do the scalar temp trick again?)
      # on-site hamiltonian block
      onsite!(H, r, R, H_nm)
      # on-site overlap matrix block (only if the model is NONORTHOGONAL)
      ISORTH || overlap!(H, M_nm)
      # add into sparse matrix
      idx = _append!(H, It, Jt, Ht, Mt, In, In, H_nm, M_nm, 1.0, NORB, idx)
   end

   # convert M, H into Sparse CCS and return
   #   NOTE: The conversion to sparse format accounts for about 1/2 of the
   #         total cost. Since It, Jt are in an ordered format, it should be
   #         possible to write a specialised code that converts it to
   #         CCS much faster, possibly with less additional allocation?
   #         another option would be to store a single It, Jt somewhere
   #         for ALL the hamiltonians, and store multiple Ht, Mt and convert
   #         these "on-the-fly", depending on whether full or sparse is needed.
   #         but at the moment, eigfact cost MUCH more than the assembly,
   #         so we could choose to stop here.
   return ISORTH ? (sparse(It, Jt, Ht), I) :
                   (sparse(It, Jt, Ht), sparse(It, Jt, Mt))
end
