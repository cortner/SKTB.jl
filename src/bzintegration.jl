
# collection functionality for BZ integration.
# For now this simply defined the Monkorst-Pack grid (uniform grid in
# BZ, due to periodicity, this yields spectral accuracy)
# TODO:
#   * loading of BZ-grid files
#   * better exploit BZ symmetries?
#   * high-accuracy adaptive integration?


# basic iterator for BZ zone integration
#  for (w, k) in quadrule
#       ...

Base.start(q::BZQuadratureRule) = (w_and_pts(q), 1)
Base.done(q::BZQuadratureRule, state) = (state[2] > length(q))
Base.next(::BZQuadratureRule, state) =
   (state[1][1][state[2]], state[1][2][state[2]]), (state[1], state[2]+1)

"""
`GammaPoint`: defines Γ-point BZ integration, i.e., a single quadrature point
in the origin.
"""
type GammaPoint <: BZQuadratureRule end

Base.length(::GammaPoint) = 1

w_and_pts(::GammaPoint) = ([1.0], [JVecF([0.0,0.0,0.0])])

"""
`MPGrid`: defines a Monkhorst-Pack grid for BZ integration

## Constructors:
```
MPGrid(cell::Matrix{Float64}, nkpoints)
MPGrid(at::AbstractAtoms, nkpoints)
```
 * `nkpoints::Tuple{Int64,Int64,Int64}`
"""
type MPGrid  <: BZQuadratureRule
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

	# We can exploit the symmetry of the BZ.
	# TODO: this is not necessarily first BZ
	#       and THE SYMMETRY HAS NOT BEEN FULLY EXPLOITED YET!!
   #       (is this a problem other than performance?)
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
