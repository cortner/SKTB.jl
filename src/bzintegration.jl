
# collection functionality for BZ integration.
# For now this simply defined the Monkorst-Pack grid (uniform grid in
# BZ, due to periodicity, this yields spectral accuracy)
# TODO:
#   * loading of BZ-grid files
#   * better exploit BZ symmetries?
#   * high-accuracy adaptive integration?




# ================ storage of k-point dependent arrays ================

"""
store k-point dependent arrays
"""
set_k_array!(at::AbstractAtoms, q, symbol, k) = set_transient!(at, (symbol, k), q)

"""
retrieve k-point dependent arrays
"""
get_k_array(at::AbstractAtoms, symbol, k) = get_transient(at, (symbol, k))

"""
check that a k-array exists
"""
has_k_array(at::AbstractAtoms, symbol, k) = has_transient(at, (symbol, k))


# ===================== Gamma-point =================


"""
`GammaPoint`: defines Γ-point BZ integration, i.e., a single quadrature point
in the origin.
"""
mutable struct GammaPoint <: BZQuadratureRule end

Base.length(::GammaPoint) = 1

w_and_pts(::GammaPoint) = ([1.0], [JVecF([0.0,0.0,0.0])])


# ================ Monkhorst-Pack Grid ===================

"""
`MPGrid`: defines a Monkhorst-Pack grid for BZ integration

## Constructors:
```
MPGrid(cell::Matrix{Float64}, nkpoints)
MPGrid(at::AbstractAtoms, nkpoints)
```
 * `nkpoints::Tuple{Int64,Int64,Int64}`
"""
mutable struct MPGrid  <: BZQuadratureRule
   k::Vector{JVecF}
   w::Vector{Float64}
end

Base.length(q::MPGrid) = length(q.w)

w_and_pts(qrule::MPGrid) = (qrule.w, qrule.k)

MPGrid(cell::Matrix{Float64}, nkpoints::Tuple{Int64, Int64, Int64}) =
   MPGrid(monkhorstpackgrid(cell, nkpoints)...)

MPGrid(at::AbstractAtoms, nkpoints::Tuple{Int64, Int64, Int64}) =
   MPGrid(cell(at), nkpoints)


"""
 `monkhorstpackgrid(cell, nkpoints)` : constructs an MP grid for the
computational cell defined by `cell` and `nkpoints`.
MonkhorstPack: K = {b/kz * j + shift}_{j=-kz/2+1,...,kz/2} with shift = 0.0.
Returns

### Parameters

* 'cell' : 3 × 1 array of lattice vector for (super)cell
* 'nkpoints' : 3 × 1 array of number of k-points in each direction. Now
it can only be (0, 0, kz::Int).

### Output

* `K`: `JVecsF` vector of length Nk
* `weights`: integration weights; scalar (uniform grid) or Nk vector.
"""
function monkhorstpackgrid(cell::Matrix{Float64},
                           nkpoints::Tuple{Int64, Int64, Int64})
   kx, ky, kz = nkpoints
   ## We need to check somewhere that 'nkpoints' and 'pbc' are compatable,
   ## e.g., if pbc[1]==false, then kx!=0 should return an error.
	# if kx != 0 || ky != 0
   #    error("This boundary condition has not been implemented yet!")
   # end
	## We want to sample the Γ-point (which is not really necessary?)
	if mod(kx,2) == 1 || mod(ky,2) == 1 || mod(kz,2) == 1
	     throw(ArgumentError("k should be an even number in Monkhorst-Pack
                grid so that the Γ-point can be sampled!"))
	end
	# compute the lattice vector of reciprocal space
   B = 2*pi*pinv(cell)
   b1, b2, b3 = JVec(B[:,1]), JVec(B[:,2]), JVec(B[:,3])

	# We can exploit the symmetry of the BZ (somewhat; more coudl be done here).
	nx, ny, nz = nn = max(kx, 1), max(ky, 1), max(kz, 1)
   kxs, kys, kzs = (kx==0 ? nx : (kx/2)), (ky==0 ? ny : (ky/2)), (kz==0 ? nz : (kz/2))
	N = nx * ny * nz
	K = zerovecs(N)
	weight = zeros(N)

	# evaluate K and weight
   #   TODO: make this loop use vectors more efficiently
   for k1 = 1:nx, k2 = 1:ny, k3 = 1:nz
      # TODO: use `sub2ind` here
      k = k1 + (k2-1) * nx + (k3-1) * nx * ny
      @assert k == sub2ind((nx, ny, nz), k1, k2, k3)
      # check when kx==0 or ky==0 or kz==0
      K[k] = (k1 - kxs) * b1/nx + (k2 - kys) * b2/ny + (k3 - kzs) * b3/nz
      weight[k] = 1.0 / ( nx * ny * nz )
   end

    return K, weight
end




# ================ BZ iterators ================

# basic iterator for BZ zone integration
#  for (w, k) in quadrule
#       ...

Base.start(q::BZQuadratureRule) = (w_and_pts(q), 1)
Base.done(q::BZQuadratureRule, state) = (state[2] > length(q))
Base.next(::BZQuadratureRule, state) =
   (state[1][1][state[2]], state[1][2][state[2]]), (state[1], state[2]+1)



# we can replace the double-loop over bz and then over the eigenvalues with
#
#   for (w, k, e, ψ) in BZiter(tbm)
#       ...
# (if ψ is a view of C_k[:, s] then this doesn't require allocation either)

"""
`BZiter`: replaces the double-loop
```
update!(at, tbm)
for (w, k) in bzquad
   for s = 1:length(w)
```
with a single-loop
```
for (w, k, ϵ, ψ) in BZiter(tbm, at)
```
Note in particular that it calls `update!` at the beginning of the
loop to precompute all the k-arrays.
"""
mutable struct BZiter
   tbm::TBModel
   at::AbstractAtoms
   w::Vector{Float64}
   k::Vector{JVecF}
end

mutable struct BZstate
   ik::Int    # index into the w, k arrays in BZ
   is::Int    # index into the epsn_k and C_k arrays in BZstate
   epsn_k::Vector{Float64}
   C_k::Matrix{Complex128}
end

BZiter(tbm::TBModel, at::AbstractAtoms) = BZiter(tbm, at, w_and_pts(tbm.bzquad)...)

Base.start(::BZiter) = BZstate(1, 1, Vector{Float64}(), Matrix{Complex128}())

Base.done(bz::BZiter, state::BZstate) = (state.ik > length(bz.w))

function Base.next(bz::BZiter, state::BZstate)
   is, ik = state.is, state.ik
   if is == 1
      if ik == 1; update!(bz.at, bz.tbm); end
      state.epsn_k = get_k_array(bz.at, :epsn, bz.k[ik])
      state.C_k = get_k_array(bz.at, :C, bz.k[ik])
   end
   # construct the next item to return to the iteration
   nextitem = (bz.w[ik], bz.k[ik], state.epsn_k[is], state.C_k[:, is])
   # update the state
   state.is += 1
   if state.is > length(state.epsn_k)
      state.is = 1
      state.ik += 1
   end
   return nextitem, state
end
