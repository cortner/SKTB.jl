# =========================  Intermediate Hamiltonian Type


typealias SKBlock{NORB} SMatrix{NORB, NORB, Float64}

"""
a triplet sparse matrix kind of thing that stores pre-computed
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
* `at` : the associated atoms object
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
   # here is a little hack that will turn an abstract type into a concrete type
   SKB = typeof(zero(SKBlock{NORB}))

   # get back positions since we still can't do without to get the cell-shifts
   # TODO: >>> maybe implement an alternative in JuLIP?
   X = positions(at)

   # ALLOCATIONS: TODO: move to a different method?
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



# import Base.-
# -(A::AbstractVector{JVecF}, a::JVecF) = JVecF[v - a for v in A]
#
# dott(a::JVecF, A::AbstractVector{JVecF}) = JVecF[dot(a, v) for v in A]
#
#
# # # append to triplet format: version 1 for H and M (non-orth TB)
# function _append!{NORB}(H::SKHamiltonian{NONORTHOGONAL, NORB},
#                   It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, idx)
#    @inbounds for i = 1:NORB, j = 1:NORB
#       idx += 1
#       It[idx] = In[i]
#       Jt[idx] = Im[j]
#       Ht[idx] = H_nm[i,j] * exp_i_kR
#       Mt[idx] = M_nm[i,j] * exp_i_kR
#    end
#    return idx
# end
#
# # append to triplet format: version 2 for H only (orthogonal TB)
# function _append!{NORB}(H::SKHamiltonian{ORTHOGONAL, NORB},
#                   It, Jt, Ht, _Mt_, In, Im, H_nm, _Mnm_, exp_i_kR, idx)
#    @inbounds for i = 1:NORB, j = 1:NORB
#       idx += 1
#       It[idx] = In[i]
#       Jt[idx] = Im[j]
#       Ht[idx] = H_nm[i,j] * exp_i_kR
#    end
#    return idx
# end
#
# # inner SKHamiltonian assembly
# #
# # NOTES:
# #  * exp_i_kR = complex multiplier needed for BZ integration
# #  * we could use cell * S instead of R[m] - (X[neigs[m]] - X[n])
# #       but this would actually be less efficient, and less clear to read
# #
# #
# function assemble!{ISORTH, NORB}(H::SKHamiltonian{ISORTH, NORB},
#                                  k, It, Jt, Ht, Mt, nlist, X)
#
#    # TODO: H_nm, M_nm, bonds could all be MArrays
#    idx = 0                     # initialise index into triplet format
#    H_nm = zeros(NORB, NORB)    # temporary arrays for computing H and M entries
#    M_nm = zeros(NORB, NORB)
#    bonds = zeros(nbonds(H))     # temporary array for storing the potentials
#
#    # loop through sites
#    for (n, neigs, r, R, _) in sites(nlist)
#       In = indexblock(n, H)     # index-block for atom index n
#       # loop through the neighbours of the current atom (i.e. the bonds)
#       for m = 1:length(neigs)
#          U = R[m]/r[m]
#          # hamiltonian block
#          sk!(H, U, hop!(H, r[m], bonds), H_nm)
#          # overlap block (only if the model is NONORTHOGONAL)
#          ISORTH || sk!(H, U, overlap!(H, r[m], bonds), M_nm)
#          # add new indices into the sparse matrix
#          Im = indexblock(neigs[m], H)
#          exp_i_kR = exp( im * dot(k, R[m] - (X[neigs[m]] - X[n])) )
#          idx = _append!(H, It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, idx)
#       end
#
#       # now compute the on-site blocks;
#       # TODO: revisit this (can one do the scalar temp trick again?) >>> only if we assume diagonal!
#       # on-site hamiltonian block
#       onsite!(H, r, R, H_nm)
#       # on-site overlap matrix block (only if the model is NONORTHOGONAL)
#       ISORTH || overlap!(H, M_nm)
#       # add into sparse matrix
#       idx = _append!(H, It, Jt, Ht, Mt, In, In, H_nm, M_nm, 1.0, idx)
#    end
#
#    # convert M, H into Sparse CCS and return
#    #   NOTE: The conversion to sparse format accounts for about 1/2 of the
#    #         total cost. Since It, Jt are in an ordered format, it should be
#    #         possible to write a specialised code that converts it to
#    #         CCS much faster, possibly with less additional allocation?
#    #         another option would be to store a single It, Jt somewhere
#    #         for ALL the hamiltonians, and store multiple Ht, Mt and convert
#    #         these "on-the-fly", depending on whether full or sparse is needed.
#    #         but at the moment, eigfact cost MUCH more than the assembly,
#    #         so we could choose to stop here.
#    return ISORTH ? (sparse(It, Jt, Ht), I) :
#                    (sparse(It, Jt, Ht), sparse(It, Jt, Mt))
# end




# ================== Intermediate Storage for dHij/dRa elements

"""
a sparse-matrix kind of thing that stores pre-computed hamiltonian
derivative blocks
"""
type SparseSKHgrad{HT, TV}
   H::HT
   at::AbstractAtoms
   i::Vector{Int32}
   j::Vector{Int32}
   first::Vector{Int32}
   dH::Vector{TV}    # derivatives of hopping blocks
   dOS::Vector{TV}   # derivatives of on-site blocks
   dM::Vector{TV}    # derivatives of off-diagonal overlap blocks
   Rcell::Vector{JVecF}
end

# typealias SKBlockGrad{NORB} SArray{Size(3, NORB, NORB), Float64}

SKBlockGradType{IO, NORB}(H::SKHamiltonian{IO, NORB}) = typeof(@SArray zeros(3, NORB, NORB))

typealias SKBg SArray{(3,4,4),Float64,3,48}

function SparseSKHgrad{ISORTH, NORB}(H::SKHamiltonian{ISORTH, NORB}, at::AbstractAtoms)
   # SKBg = SKBlockGradType(H)
   # TODO: >>> maybe implement an alternative in JuLIP?
   X = positions(at)

   # ALLOCATIONS: TODO: move to a different method?
   # neighbourlist
   nlist = neighbourlist(at, cutoff(H))
   #      off-diagonal + diagonal
   nnz = length(nlist)
   # allocate space for ordered triplet format
   i = zeros(Int32, nnz)
   j = zeros(Int32, nnz)
   first = zeros(Int32, length(at))
   dH = zeros(SKBg, nnz)
   dOS = zeros(SKBg, nnz)
   dM = ISORTH ? Vector{SKBg}() : zeros(SKBg, nnz)
   Rcell = zeros(JVecF, nnz)
   # index into the triplet format
   idx = 0

   # allocate work arrays
   maxneigs = maximum(length(jj) for (_1, jj, _2, _3, _4) in sites(nlist))
   const dH_nn = zeros(3, NORB, NORB, maxneigs)
   const dH_nm = zeros(3, NORB, NORB)
   const dM_nm = zeros(3, NORB, NORB)
   const bonds = zeros(nbonds(H))
   const dbonds = zeros(nbonds(H))

   for (n, neigs, r, R, _) in sites(nlist)
      first[n] = idx+1

      # onsite derivative; this will be copied into the new format in the
      # following loop
      onsite_grad!(H, r, R, dH_nn)

      # loop through neighbours
      for i_n = 1:length(neigs)
         m = neigs[i_n]

         # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
         hop_d!(H, r[i_n], bonds, dbonds)
         sk_d!(H, r[i_n], -R[i_n], bonds, dbonds, dH_nm)
         if !ISORTH
            overlap_d!(H, r[i_n], bonds, dbonds)
            sk_d!(H, r[i_n], -R[i_n], bonds, dbonds, dM_nm)
         end

         # store in new format
         idx += 1
         i[idx], j[idx], dH[idx] = n, m, SKBg(dH_nm)
         if !ISORTH; dM[idx] = SKBg(dM_nm); end
         dOS[idx] = SKBg(view(dH_nn, :, :, :, i_n))
         Rcell[idx] = R[i_n] - (X[m] - X[n])
      end
   end

   # # TEST
   # for n = 1:10
   #    ii, jj, dh = i[n], j[n], dH[n]
   #    @show dh
   #    # R = X[jj] - X[ii]
   #    # r = norm(R)
   #    # hop_d!(H, r, bonds, dbonds)
   #    # sk_d!(H, r, -R, bonds, dbonds, dH_nm)
   #    # @assert vecnorm(dH_nm - dh, Inf) == 0.0
   #    # @show vecnorm(dH_nm - dh,Inf)
   # end


   return SparseSKHgrad(H, at, i, j, first, dH, dOS, dM, Rcell)
end
