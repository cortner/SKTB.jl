
import JuLIP.Potentials: evaluate, evaluate_d

using ForwardDiff

# slaterkoster.jl
#
# Collect all generic stuff for Slater-Koster-type Tight-binding
# models (which is 99.99% of non-selfconsistent TB models)
#

abstract SKHamiltonian{ISORTH, NORB} <: TBHamiltonian{ISORTH}

norbitals{ISORTH,NORB}(::SKHamiltonian{ISORTH, NORB}) = NORB

nbonds{ISORTH}(::SKHamiltonian{ISORTH, 1}) = 1
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 4}) = 4
nbonds{ISORTH}(::SKHamiltonian{ISORTH, 9}) = 10

ndofs(H::SKHamiltonian, at::AbstractAtoms) = norbitals(H) * length(at)

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

######## s-orbital model

function sk!{IO}(H::SKHamiltonian{IO, 1}, U, bonds, out)
   setindex!(out, bonds[1], 1)
   return out
end

function sk_d!{IO}(H::SKHamiltonian{IO, 1}, r, R, b, db, dH_nm)
   # dH_nm is 3 x 1 x 1 so we can just index it linearly    (NORB = 1)
   for a = 1:3
      dH_nm[a] = db[1] * R[a] / r
   end
   return dH_nm
end

######## sp-orbital model

sk!{IO}(H::SKHamiltonian{IO, 4}, U, bonds, out) = _sk4!(U, bonds, out)

sk_d!{IO}(H::SKHamiltonian{IO, 4}, r, R, b, db, dout) = _sk4_d!(R/r, r, b, db, dout)

######## spd-orbital model

sk!{IO}(H::SKHamiltonian{IO, 9}, U, bonds, out) = _sk9!(U, bonds, out)



function sk_ad!{IO, NORB}(H::SKHamiltonian{IO, NORB}, r, R, b, db, dout)
   A = zeros(ForwardDiff.Dual{3,Float64}, NORB, NORB)
   f = S -> sk!(H, S / norm(S), adhop(H, norm(S)), A)
   dA = ForwardDiff.jacobian(f, Vector(R))
   dA = reshape(dA, NORB, NORB, 3)
   for a = 1:3, i = 1:NORB, j = 1:NORB
      dout[a, i, j] = dA[i, j, a]
   end
   return dout
end



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

@protofun hop(::SKHamiltonian, ::Any, ::Any)

hop_d(H::SKHamiltonian, r, i) = ForwardDiff.derivative(s -> hop(H,s,i), r)

function hop!(H::SKHamiltonian, r, bonds)
   for i = 1:nbonds(H)
      bonds[i] = hop(H, r, i)
   end
   return bonds
end

adhop(H::SKHamiltonian, r) = [hop(H, r, i) for i = 1:nbonds(H)]

function hop_d!(H::SKHamiltonian, r, b, db)
   for i = 1:nbonds(H)
      b[i] = hop(H, r, i)
      db[i] = hop_d(H, r, i)
   end
   return b, db
end

@protofun overlap(::SKHamiltonian, ::Real, ::Integer)

overlap_d(H::SKHamiltonian, r::Real, i) = ForwardDiff.derivative(s->overlap(H,s,i), r)

function overlap!(H::SKHamiltonian, r, bonds)
   for i = 1:nbonds(H)
      bonds[i] = overlap(H, r, i)
   end
   return bonds
end

function overlap_d!(H::SKHamiltonian, r, b, db)
   for i = 1:nbonds(H)
      b[i] = overlap(H, r, i)
      db[i] = overlap_d(H, r, i)
   end
   return b, db
end


function onsite! end
function onsite_grad! end

# # prototypes for the defaul diagonal on-site terms
# function diagonsite! end
# function diagonsite_d! end
#
# function onsite!(H::SKHamiltonian, r, R, H_nn)
#    diagonsite!(H, r, H_nn)   # this treats H_nn as linear memory
#    for i = 1:size(H_nn, 1)
#       H_nn[i,i] = H_nn[i]    # hence H_nn[i] must be copied to the diagonal
#       H_nn[i] = 0.0
#    end
#    return H_nn
# end

# function onsite_grad!(H::SKHamiltonian, r, dH_nn)
#    temp = zeros(
#
# end




# inner SKHamiltonian assembly
#
# NOTES:
#  * exp_i_kR = complex multiplier needed for BZ integration
#  * we could use cell * S instead of R[m] - (X[neigs[m]] - X[n])
#       but this would actually be less efficient, and less clear to read
#
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
         # hamiltonian block
         # sk!(H, U, hop!(H, r[m], bonds), H_nm)
         sk!(H, U, adhop(H, r[m]), H_nm)
         # overlap block (only if the model is NONORTHOGONAL)
         ISORTH || sk!(H, U, overlap!(H, r[m], bonds), M_nm)
         # add new indices into the sparse matrix
         Im = indexblock(neigs[m], H)
         exp_i_kR = exp( im * dot(k, R[m] - (X[neigs[m]] - X[n])) )
         idx = _append!(H, It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, NORB, idx)
      end

      # now compute the on-site blocks;
      # TODO: revisit this (can one do the scalar temp trick again?) >>> only if we assume diagonal!
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



# ============ SPECIALISED SK FORCES IMPLEMENTATION =============


#
# this is an old force computation that *requires* a SKHamiltonian structure
#  therefore, I added H as an argument so _forces_k dispatches on its type
#
# F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
#
# TODO: rewrite this in a type-agnostic way!
#      (but probably keep this version for performance? or at least comparison?)
#
function _forces_k{ISORTH, NORB}(X::JVecsF, at::AbstractAtoms, tbm::TBModel,
                  H::SKHamiltonian{ISORTH,NORB}, nlist, k::JVecF)
   # obtain the precomputed arrays
   epsn = get_k_array(at, :epsn, k)::Vector{Float64}
   C = get_k_array(at, :C, k)::Matrix{Complex128}
   # df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))
   df = grad(tbm.smearing, epsn)

   # precompute some products
   const C_df_Ct = (C * (df' .* C)')               #::Matrix{Complex{Float64}}
   const C_dfepsn_Ct = (C * ((df.*epsn)' .* C)')   #::Matrix{Complex{Float64}}

   # allocate array for forces
   const frc = zeros(Complex{Float64}, 3, length(X))

   # pre-allocate dH, with a (dumb) initial guess for the size
   # TODO:
   #   * re-interpret these as arrays of JVecs, where the first argument is the JVec
   #   * move them outside to prevent multiple allocations
   const dH_nn = zeros(3, NORB, NORB, 6)
   const dH_nm = zeros(3, NORB, NORB)
   const dM_nm = zeros(3, NORB, NORB)

   const bonds = zeros(nbonds(H))
   const dbonds = zeros(nbonds(H))

   # loop through all atoms, to compute the force on atm[n]
   for (n, neigs, r, R, _) in sites(nlist)
      # neigs::Vector{Int}   # TODO: put this back in?!?  > PROFILE IT AGAIN
      # R::Matrix{Float64}  # TODO: put this back in?!?
      #   IT LOOKS LIKE the type of R might not be inferred!

      # compute the block of indices for the orbitals belonging to n
      In = indexblock(n, H)

      # compute ∂H_mm/∂y_n (onsite terms) M_nn = const ⇒ dM_nn = 0
      # dH_nn should be 3 x norbitals x norbitals x nneigs
      if length(neigs) > size(dH_nn, 4)
         dH_nn = zeros(3, NORB, NORB, ceil(Int, 1.5*length(neigs)))
      end
      onsite_grad!(H, r, R, dH_nn)

      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, H)
         kR = dot(R[i_n] - (X[m] - X[n]), k)
         eikr = exp(im * kR)::Complex{Float64}

         # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
         # grad!(tbm.hop, r[i_n], - R[i_n], dH_nm)
         # hop_d!(H, r[i_n], bonds, dbonds)
         sk_ad!(H, r[i_n], -R[i_n], bonds, dbonds, dH_nm)

         # grad!(tbm.overlap, r[i_n], - R[i_n], dM_nm)
         if !ISORTH
            overlap_d!(H, r[i_n], bonds, dbonds)
            sk_d!(H, r[i_n], -R[i_n], bonds, dbonds, dM_nm)
         end

         # the following is a hack to put the on-site assembly into the
         # innermost loop
         # F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
         for a = 1:NORB, b = 1:NORB
            t1 = 2.0 * real(C_df_Ct[Im[a], In[b]] * eikr)
            t2 = 2.0 * real(C_dfepsn_Ct[Im[a], In[b]] * eikr)
            t3 = C_df_Ct[In[a],In[b]]
            # add contributions to the force
            # TODO: can re-write this as sum over JVecs
            for j = 1:3
               frc[j,n] += -dH_nm[j,a,b] * t1 + dM_nm[j,a,b] * t2 + dH_nn[j,a,b,i_n] * t3
               frc[j,m] -= t3 * dH_nn[j,a,b,i_n]
            end
         end

      end  # m in neigs-loop
   end  #  sites-loop

   # TODO: in the future assemble the forces already in JVecsF format
   return real(frc) |> vecs
end


# imported from JuLIP
function forces(tbm::TBModel, atm::AbstractAtoms)
   update!(atm, tbm)
   nlist = neighbourlist(atm, cutoff(tbm.H))
   frc = zerovecs(length(atm))
   X = positions(atm)
   for (w, k) in tbm.bzquad
      frc +=  w * _forces_k(X, atm, tbm, tbm.H, nlist, k)
   end
   return frc
end
