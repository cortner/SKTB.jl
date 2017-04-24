import JuLIP.Potentials: evaluate, evaluate_d

using ForwardDiff

# slaterkoster.jl
#
# Collect all generic stuff for Slater-Koster-type Tight-binding
# models (which is 99.99% of non-selfconsistent TB models)
#



"""
`indexblock`:
a little auxiliary function to compute indices of Slater Koster orbitals,
this is returned as an SVector, i.e. it is generated on the stack so that no
heap memory is allocated.
"""
indexblock{IO,NORB}(n::Integer, H::SKHamiltonian{IO,NORB}) =
   SVector{NORB, Int}( ((n-1)*NORB+1):(n*NORB) )

function indexblock{T <: Integer}(Iat::AbstractVector{T}, H::SKHamiltonian)
   out = T[]
   for n in Iat
      append!(out, indexblock(n, H))
   end
   return out
end


############################################################
### Hamiltonian entries


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
sk_d!{IO}(H::SKHamiltonian{IO, 9}, r, R, b, db, dout) = _sk9_d!(R/r, r, b, db, dout)


# ======================================================================
# prototypes for the functions needed to assemble the hamiltonian
# and its derivatives

# TODO: documentation, what is this called?

"""
`hop(H::SKHamiltonian, r, i)`: where `r` is real, `i` integer, this
should return the
"""
@protofun hop(::SKHamiltonian, ::Any, ::Any)


# TODO: this is probably type-unstable!
hop_d(H::SKHamiltonian, r, i) = ForwardDiff.derivative(s -> hop(H,s,i), r)


function hop!(H::SKHamiltonian, r, bonds)
   for i = 1:nbonds(H)
      bonds[i] = hop(H, r, i)
   end
   return bonds
end

# TODO: hop_d should return hop and hop_d, since we need both!!!!
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

# rewrite onsite! as diagonals???
# but we don't provide AD for these since they depend on too many variables,
# so AD will necessarily be inefficient.

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
   if has_transient(at, :SKH)
      return get_transient(at, :SKH)
   end

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

   skh = SparseSKH(H, at, i, j, first, vH, vM, Rcell)
   set_transient!(at, :SKH, skh)
   return skh
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
   Hout[diagind(Hout)] = real(Hout[diagind(Hout)])
   Mout[diagind(Mout)] = real(Mout[diagind(Mout)])
   return Hout, Mout
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
   return Hout, I
end



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


#
# this is a work-around until we move to v0.6; at that point
# StaticArrays changes and we can do
#    typealias SKBlockGrad{NORB} SArray{Tuple{3,NORB,NORB}, ...}
# instead. But there is still the problem with D and L
#
SKBlockGradType{IO, NORB}(H::SKHamiltonian{IO, NORB}) = typeof(@SArray zeros(3, NORB, NORB))


function SparseSKHgrad{ISORTH, NORB}(H::SKHamiltonian{ISORTH, NORB}, at::AbstractAtoms)
   # if the array has previously been computed, just return it
   if has_transient(at, :SKBg)
      return get_transient(at, :SKBg)
   end
   # if not generate it from scratch

   # entry-type
   SKBg = SKBlockGradType(H)

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

      # onsite derivative; copied into the new format in the following loop
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

   skhg = SparseSKHgrad(H, at, i, j, first, dH, dOS, dM, Rcell)
   set_transient!(at, :SKHg, skhg)
   return skhg
end
