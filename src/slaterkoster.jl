
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

"""
`indexblock`:
a little auxiliary function to compute indices of Slater Koster orbitals,
this is returned as an SVector, i.e. it is generated on the stack so that no
heap memory is allocated.
"""
indexblock{IO,NORB}(n::Integer, H::SKHamiltonian{IO,NORB}) =
   SVector{NORB, Int}( ((n-1)*NORB+1):(n*NORB) )


# function skindexblock{T <: Integer}(Iat::AbstractVector{T}, norb::Integer)
#    out = T[]
#    for n in Iat
#       append!(out, indexblock(n, norb))
#    end
#    return out
# end
#
# indexblock{T <: Integer}(Iat::AbstractVector{T}, H::SKHamiltonian) =
#    skindexblock(Iat, norbitals(H))





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



function sk_ad!{IO, NORB}(H::SKHamiltonian{IO, NORB}, R, bfun, dout)
   A = zeros(ForwardDiff.Dual{3, Float64}, NORB, NORB)
   dA = zeros(NORB*NORB, 3)
   f = S -> sk!(H, S / norm(S), [bfun(H, norm(S), i) for i = 1:nbonds(H)], A)
   dA = ForwardDiff.jacobian!(dA, f, Vector{Float64}(R))
   dB = reshape(dA, NORB, NORB, 3)
   for a = 1:3, i = 1:NORB, j = 1:NORB
      dout[a, i, j] = dB[i, j, a]
   end
   return dout
end


function bonds_test!{IO, NORB}(H::SKHamiltonian{IO, NORB}, r, bfun, bonds)
   for i = 1:NORB
      bonds[i] = bfun(H, r, i)
   end
   return bonds
end

function sk_ad_test!{IO, NORB}(H::SKHamiltonian{IO, NORB}, R, bfun, dout, A, dA, bonds)
   f = S -> sk!(H, S / norm(S), bonds_test!(H, norm(S), bfun, bonds), A)
   Rv = Vector{Float64}(R)
   ForwardDiff.jacobian!(dA, f, Rv)
   dB = reshape(dA, NORB, NORB, 3)
   for a = 1:3, i = 1:NORB, j = 1:NORB
      dout[a, i, j] = dB[i, j, a]
   end
   return dout
end


# adhop(H::SKHamiltonian, r) = [hop(H, r, i) for i = 1:nbonds(H)]


# prototypes for the functions needed in `assemble!`

@protofun hop(::SKHamiltonian, ::Any, ::Any)

hop_d(H::SKHamiltonian, r, i) = ForwardDiff.derivative(s -> hop(H,s,i), r)

function hop!(H::SKHamiltonian, r, bonds)
   for i = 1:nbonds(H)
      bonds[i] = hop(H, r, i)
   end
   return bonds
end


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
function _append!{NORB}(H::SKHamiltonian{NONORTHOGONAL, NORB},
                  It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, idx)
   @inbounds for i = 1:NORB, j = 1:NORB
      idx += 1
      It[idx] = In[i]
      Jt[idx] = Im[j]
      Ht[idx] = H_nm[i,j] * exp_i_kR
      Mt[idx] = M_nm[i,j] * exp_i_kR
   end
   return idx
end

# append to triplet format: version 2 for H only (orthogonal TB)
function _append!{NORB}(H::SKHamiltonian{ORTHOGONAL, NORB},
                  It, Jt, Ht, _Mt_, In, Im, H_nm, _Mnm_, exp_i_kR, idx)
   @inbounds for i = 1:NORB, j = 1:NORB
      idx += 1
      It[idx] = In[i]
      Jt[idx] = Im[j]
      Ht[idx] = H_nm[i,j] * exp_i_kR
   end
   return idx
end



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
         sk!(H, U, hop!(H, r[m], bonds), H_nm)
         # overlap block (only if the model is NONORTHOGONAL)
         ISORTH || sk!(H, U, overlap!(H, r[m], bonds), M_nm)
         # add new indices into the sparse matrix
         Im = indexblock(neigs[m], H)
         exp_i_kR = exp( im * dot(k, R[m] - (X[neigs[m]] - X[n])) )
         idx = _append!(H, It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, idx)
      end

      # now compute the on-site blocks;
      # TODO: revisit this (can one do the scalar temp trick again?) >>> only if we assume diagonal!
      # on-site hamiltonian block
      onsite!(H, r, R, H_nm)
      # on-site overlap matrix block (only if the model is NONORTHOGONAL)
      ISORTH || overlap!(H, M_nm)
      # add into sparse matrix
      idx = _append!(H, It, Jt, Ht, Mt, In, In, H_nm, M_nm, 1.0, idx)
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
#    F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
# but instead of computing H,n, M,n as matrices, this assembly loops over
# the non-zero blocks first and the inner loop is over the derivative ,n.
#
# TODO: after rewriting the generic force assembly, compare against this
#       implementation for performance
#
function _forces_k{ISORTH, NORB}(X::JVecsF, at::AbstractAtoms, tbm::TBModel,
                                 H::SKHamiltonian{ISORTH,NORB}, nlist, k::JVecF)
   # obtain the precomputed arrays
   epsn = get_k_array(at, :epsn, k)::Vector{Float64}
   C = get_k_array(at, :C, k)::Matrix{Complex128}
   # df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))
   df = grad(tbm.smearing, epsn)::Vector{Float64}

   # precompute some products
   C_df_Ct = (C * (df' .* C)')
   C_dfepsn_Ct = (C * ((df.*epsn)' .* C)')

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
         # kR is actually independent of X, so need not be differentiated
         kR = dot(R[i_n] - (X[m] - X[n]), k)
         eikr = exp(im * kR)::Complex{Float64}

         # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
         hop_d!(H, r[i_n], bonds, dbonds)
         sk_d!(H, r[i_n], -R[i_n], bonds, dbonds, dH_nm)
         if !ISORTH
            overlap_d!(H, r[i_n], bonds, dbonds)
            sk_d!(H, r[i_n], -R[i_n], bonds, dbonds, dM_nm)
         end

         # the following is a hack to put the on-site assembly into the innermost loop
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


# * `forces` is imported from JuLIP
# * this implementation is the old version from Atoms.jl, which makes
#   specific assumptions about the structure of the hamiltonian, hence is
#   only valid for SK-type hamiltonians.
#
# TODO: after implementing the generic force assembly, we need to benchmark
#       them against each other!
function forces{HT <: SKHamiltonian}(tbm::TBModel{HT}, atm::AbstractAtoms)
   update!(atm, tbm)
   nlist = neighbourlist(atm, cutoff(tbm.H))
   frc = zerovecs(length(atm))
   X = positions(atm)
   for (w, k) in tbm.bzquad
      frc +=  w * _forces_k(X, atm, tbm, tbm.H, nlist, k)
   end
   return frc
end




# =========================  forces and other kinds of derivatives


typealias SKBlock{NORB} SMatrix{NORB, NORB, Float64}

"""
a sparse matrix kind of thing that stores pre-computed
hamiltonian blocks, and can efficiently generate k-dependent Hamiltonians,
either sparse of full.

### Methods:
* `full(A::SparseSKH, k)`: returns full `H`, `M` for given `k`-vector.
* `full!(out, A, k)`: same as `full`, but in-place

### Methods to be implemented:
* `sparse(A::SparseSKH, k)`: same but sparse
* `collect(A::SparseSKH, k)`: chooses full (small systems) or sparse (large
                           systems) based on some simple heuristic

### Fields:
* `H` : the hamiltonian used to construct it
* `i, j` : row and column indices
* `first` : `first[n]` is the index in `i, j` for which `i[idx] == n`
* `vH` : Hamiltonian blocks
* `vM` : overlap blocks
* `Rcell` : each hamiltonian block is associated with an e^{i k ⋅ S} multiplier
            from the Bloch transform; this S is stored in Rcell.
"""
immutable SparseSKH{HT, TV}  # v0.6: require that TV <: SKBlock{NORB}
   H::HT
   at::AbstractAtoms
   i::Vector{Int32}
   j::Vector{Int32}
   first::Vector{Int32}
   vH::Vector{TV}
   vM::Vector{TV}
   Rcell::Vector{JVecF}
end

function SparseSKH{ISORTH, NORB}(H::SKHamiltonian{ISORTH, NORB}, at::AbstractAtoms)
   SKB = typeof(zero(SKBlock{NORB}))
   X = positions(at)

   # neighbourlist
   nlist = neighbourlist(at, cutoff(H))
   #      off-diagonal + diagonal
   nnz = length(nlist) + length(at)
   # allocate space for ordered triplet format
   i = zeros(Int32, nnz)
   j = zeros(Int32, nnz)
   first = zeros(Int32, length(at))
   vH = zeros(SKB, nnz)
   vM = ISORTH ? Vector{SKB}() : zeros(SKB, nnz)
   Rcell = zeros(JVecF, nnz)
   # index into the triplet format
   idx = 0

   # allocate space to assemble the hamiltonian blocks, we use MMatrix
   # here, but is this really necessary? Matrix should do fine?
   H_nm = zero(MMatrix{NORB, NORB, Float64})
   M_nm = zero(MMatrix{NORB, NORB, Float64})
   bonds = zeros(nbonds(H))     # temporary array for storing the potentials

   # loop through sites
   for (n, neigs, r, R, _) in sites(nlist)
      first[n] = idx+1          # where do the triplet entries for atom n start?

      # add the diagonal/on-site entries
      onsite!(H, r, R, H_nm)
      idx += 1
      i[idx], j[idx], vH[idx] = n, n, SKB(H_nm)
      # on-site overlap matrix block (only if the model is NONORTHOGONAL)
      if !ISORTH
         overlap!(H, M_nm)
         vM[idx] = SKB(M_nm)
      end
      # compute the Rvector corresponding to this block
      Rcell[idx] = zero(JVecF)

      # loop through the neighbours of the current atom (i.e. the bonds)
      for m = 1:length(neigs)
         U = R[m]/r[m]
         # hamiltonian block
         sk!(H, U, hop!(H, r[m], bonds), H_nm)
         idx += 1
         i[idx], j[idx], vH[idx] = n, neigs[m], SKB(H_nm)
         # overlap block (only if the model is NONORTHOGONAL)
         if !ISORTH
            sk!(H, U, overlap!(H, r[m], bonds), M_nm)
            vM[idx] = SKB(M_nm)
         end
         # compute the Rcell vector for these blocks
         Rcell[idx] = R[m] - (X[neigs[m]] - X[n])
      end
   end
   return SparseSKH(H, at, i, j, first, vH, vM, Rcell)
end

_alloc_full(skh::SparseSKH) = _alloc_full(skh.H, skh.at)

_alloc_full(H::SKHamiltonian{NONORTHOGONAL}, at::AbstractAtoms) =
   Matrix{Complex128}(ndofs(H, at), ndofs(H, at)), Matrix{Complex128}(ndofs(H, at), ndofs(H, at))

_alloc_full(H::SKHamiltonian{ORTHOGONAL}, at::AbstractAtoms) =
   Matrix{Complex128}(ndofs(H, at), ndofs(H, at)), Matrix{Complex128}()

Base.full(H::SparseSKH, k::AbstractVector = zero(JVecF)) =
   full!(_alloc_full(H), H, k)

full!(out, H::SparseSKH, k::AbstractVector = zero(JVecF)) =
   _full!(out[1], out[2], H, k, H.H)

function _full!{NORB}(Hout, Mout, skh, k, H::SKHamiltonian{NONORTHOGONAL, NORB})
   fill!(Hout, 0.0)
   fill!(Mout, 0.0)
   k = JVecF(k)
   for (i, j, H_ij, M_ij, S) in zip(skh.i, skh.j, skh.vH, skh.vM, skh.Rcell)
      eikR = exp( im * dot(k, S) )
      Ii, Ij = indexblock(i, H), indexblock(j, H)
      @inbounds for a = 1:NORB, b = 1:NORB
         Hout[Ii[a], Ij[b]] += H_ij[a, b] * eikR
         Mout[Ii[a], Ij[b]] += M_ij[a, b] * eikR
      end
   end
   # TODO: this is a HACK; need to do it in-place
   Hout[diagind(Hout)] = real(Hout[diagind(Hout)])
   Mout[diagind(Mout)] = real(Mout[diagind(Mout)])
   return Hermitian(Hout), Hermitian(Mout)
end


function _full!{NORB}(Hout, _Mout_, skh, k, H::SKHamiltonian{ORTHOGONAL, NORB})
   fill!(Hout, 0.0)
   k = JVecF(k)
   for (i, j, H_ij, S) in zip(skh.i, skh.j, skh.vH, skh.Rcell)
      eikR = exp( im * dot(k, S) )
      Ii, Ij = indexblock(i, H), indexblock(j, H)
      @inbounds for a = 1:NORB, b = 1:NORB
         Hout[Ii[a], Ij[b]] += H_ij[a, b] * eikR
      end
   end
   Hout[diagind(Hout)] = real(Hout[diagind(Hout)])
   return Hermitian(Hout), I
end




# """
# a sparse-matrix kind of thing that stores pre-computed hamiltonian
# derivative blocks
# """
# type SparseGradSKH
#
# end
#
#
# """
# `pre_derivatives(H::SKHamiltonian, at::AbstractAtoms)`
#
# precomputes all the hamiltonian derivatives
# """
# function pre_derivatives!(H::SKHamiltonian, at::AbstractAtoms)
#
# end
