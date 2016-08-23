
using JuLIP
using JuLIP.Potentials

using FixedSizeArrays

import JuLIP: energy, forces
import JuLIP.Potentials: cutoff, @pot, evaluate, evaluate_d


export hamiltonian, densitymatrix


# TODO: making SmearingFunction a potential is a bit of a hack:?
#       or is it? It is energy after all?
abstract SmearingFunction <: Potential


# TODO: default evaluate!; should this potentially go into Potentials?
evaluate!(pot, r, R, target)  = copy!(target, evaluate(pot, r, R))
evaluate_d!(pot, r, R, target)  = copy!(target, evaluate_d(pot, r, R))
grad(pot, r, R) = R .* (evaluate_d(pot, r, R) ./ r)'
grad!(p, r, R, G) = copy!(G, grad(p, r, R))


"""
`TBModel`: basic non-self consistent tight binding calculator. This
type admits TB models where the Hamiltonian is of the form

    H_ij = f_hop(r_ij)   if i ≠ j
    H_ii = f_os( {r_ij}_j )

i.e. the hopping terms is a pair potential while the on-site terms are
more general; this is consistent in particular with the NRL TB model.
This model can only descibe a single species of atoms.
"""
type TBModel{P_os, P_hop, P_ol, P_p} <: AbstractTBModel
    # Hamiltonian parameters
    onsite::P_os
    hop::P_hop
    overlap::P_ol
    # repulsive potential
    pair::P_p

    # HJ: add a parameter Rcut
    # since the functions "cutoff" in Potentials.jl and NRLTB.jl may conflict
    # TODO: this should not happen; need to resolve this issue
    Rcut::Float64

    # remaining model parameters
    smearing::SmearingFunction
    norbitals::Int

    #  WHERE DOES THIS GO?
    #  morally this is really part of the smearing function
    fixed_eF::Bool
    eF::Float64
    # beta::Float64

    # k-point sampling information:
    #    0 = open boundary
    #    1 = Gamma point
    nkpoints::Tuple{Int, Int, Int}

    # internals
    hfd::Float64           # step used for finite-difference approximations
    needupdate::Bool       # tells whether hamiltonian and spectrum are up-to-date
    arrays::Dict{Any, Any}      # storage for various
                                 # TODO: this ought to go into Atoms
end


typealias TightBindingModel TBModel


TBModel(;onsite = ZeroSitePotential(),
        hop = ZeroPairPotential(),
        overlap = ZeroPairPotential(),
        pair = ZeroPairPotential(),
	     Rcut = 0.0,
        smearing = ZeroTemperature(),
        norbitals = 0,
        fixed_eF = true,
        eF = 0.0,
        nkpoints = (0,0,0),
        hfd = 0.0,
        needupdate = true,
        arrays = Dict()) =
   TBModel(onsite, hop, overlap, pair, Rcut, smearing, norbitals,
           fixed_eF, eF, nkpoints, hfd, needupdate, arrays)


"""
`indexblock`:
a little auxiliary function to compute indices for several orbitals
"""
indexblock{T <: Integer}(n::T, tbm::TBModel) =
      Vec( T[(n-1) * tbm.norbitals + j for j = 1:tbm.norbitals] )

cutoff(tbm::TBModel) = tbm.Rcut
# HJ: not sure this returns right Rcut for NRL ----------------------------------
# max(cutoff(tbm.hop), cutoff(tbm.onsite), cutoff(tbm.pair))
# -------------------------------------------- ----------------------------------


# """`FermiDiracSmearing`:
#
# f(e) = ( 1 + exp( beta (e - eF) ) )^{-1}
# """

@pot type FermiDiracSmearing <: SmearingFunction
    beta
    eF
end

FermiDiracSmearing(beta;eF=0.0) = FermiDiracSmearing(beta, eF)

# FD distribution and its derivative. both are vectorised implementations
# TODO: rewrite using auto-diff!!!
#       but this requires us to admit parameters for auto-diff
fermi_dirac(eF, beta, epsn) =
    2.0 ./ ( 1.0 + exp((epsn - eF) * beta) )
fermi_dirac_d(eF, beta, epsn) =
    - fermi_dirac(eF, beta, epsn).^2 .* exp((epsn - eF) * beta) * beta / 2.0
fermi_dirac_d2(eF, beta, epsn) =
    fermi_dirac(eF, beta, epsn).^3 .* exp((epsn - eF) * 2.0 * beta) * beta^2 / 2.0 -
    fermi_dirac(eF, beta, epsn).^2 .* exp((epsn - eF) * beta) * beta^2 / 2.0
fermi_dirac_d3(eF, beta, epsn) =
    - 12.0 * fermi_dirac(eF, beta, epsn).^4 .* exp((epsn - eF) * 3.0 * beta) * beta^3 / 16.0 +
    12.0 * fermi_dirac(eF, beta, epsn).^3 .* exp((epsn - eF) * 2.0 * beta) * beta^3 / 8.0 -
    fermi_dirac(eF, beta, epsn).^2 .* exp((epsn - eF) * beta) * beta^3 / 2.0

# Boilerplate to work with the FermiDiracSmearing type
evaluate(fd::FermiDiracSmearing, epsn) = fermi_dirac(fd.eF, fd.beta, epsn)
evaluate_d(fd::FermiDiracSmearing, epsn) = fermi_dirac_d(fd.eF, fd.beta, epsn)
# Boilerplate to work with the FermiDiracSmearing type
evaluate(fd::FermiDiracSmearing, epsn, eF) = fermi_dirac(eF, fd.beta, epsn)
evaluate_d(fd::FermiDiracSmearing, epsn, eF) = fermi_dirac_d(eF, fd.beta, epsn)

function set_eF!(fd::FermiDiracSmearing, eF)
   fd.eF = eF
end


# """`ZeroTemperature`:
#
# TODO: write documentation
# """

@pot type ZeroTemperature <: SmearingFunction
   eF
end

function full_hermitian(A)
   A = 0.5 * (A + A')
   A[diagind(A)] = real(A[diagind(A)])
   return Hermitian(full(A))
end


"""`sorted_eig`:  helper function to compute eigenvalues, then sort them
in ascending order and sort the eig-vectors as well."""
function sorted_eig(H, M)
   epsn, C = eig(full_hermitian(H), full_hermitian(M))
   Isort = sortperm(epsn)
   return epsn[Isort], C[:, Isort]
end

import Base.setindex!
setindex!(tbm::TBModel, val, symbol) = set_array!(tbm, symbol, val)

import Base.getindex
getindex(tbm::TBModel, symbol) = get_array(tbm, symbol)

"store an array"
function set_array!(tbm::TBModel, key, val)
    tbm.arrays[key] = val
end

"""retrieve an array; instead of raising an exception if `key` does not exist,
 this function returns `nothing`"""
function get_array(tbm::TBModel, key)
    if haskey(tbm.arrays, key)
        return tbm.arrays[key]
    else
        return nothing
    end
end

"""store k-point dependent arrays"""
set_k_array!(tbm, q, symbol, k) = set_array!(tbm, (symbol, k), q)

"""retrieve k-point dependent arrays"""
get_k_array(tbm, symbol, k) = get_array(tbm, (symbol, k))


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


"""
`monkhorstpackgrid(atm::AbstractAtoms, tbm::TBModel)` : extracts cell and grid
information and returns an MP grid.
"""
monkhorstpackgrid(atm::AbstractAtoms, tbm::TBModel) =
                     monkhorstpackgrid(cell(atm), tbm.nkpoints)



############################################################
##### update functions


"""
`update_eig!(atm::AbstractAtoms, tbm::TBModel)` : updates the hamiltonians
and spectral decompositions on the MP grid.
"""
function update_eig!(atm::AbstractAtoms, tbm::TBModel)
    K, weight = monkhorstpackgrid(atm, tbm)
    nlist = neighbourlist(atm, cutoff(tbm))
    nnz_est = length(nlist) * tbm.norbitals^2 + length(atm) * tbm.norbitals^2
    It = zeros(Int32, nnz_est)
    Jt = zeros(Int32, nnz_est)
    Ht = zeros(Complex{Float64}, nnz_est)
    Mt = zeros(Complex{Float64}, nnz_est)
    X = positions(atm)
    for n = 1:size(K, 2)
        k = K[n]
        H, M = hamiltonian!(tbm, k, It, Jt, Ht, Mt, nlist, X)
        epsn, C = sorted_eig(H, M)
        set_k_array!(tbm, epsn, :epsn, k)
        set_k_array!(tbm, C, :C, k)
    end
end


"""
`update!(atm::AbstractAtoms, tbm:TBModel)`: checks whether the precomputed
data stored in `tbm` needs to be updated (by comparing atom positions) and
if so, does all necessary updates. At the moment, the following are updated:

* spectral decompositions (`update_eig!`)
* the fermi-level (`update_eF!`)
"""
function update!(atm::AbstractAtoms, tbm::TBModel)
   Xnew = positions(atm)
   Xold = tbm[:X]   # (returns nothing if X has not been stored previously)
   if Xnew != Xold
      tbm[:X] = copy(Xnew)
      # do all the updates
      update_eig!(atm, tbm)
      update_eF!(atm, tbm)
   end
end


"""
`update_eF!(tbm::TBModel)`: recompute the correct
fermi-level; using the precomputed data in `tbm.arrays`
"""
function update_eF!(atm::AbstractAtoms, tbm::TBModel)
   if tbm.fixed_eF
      set_eF!(tbm.smearing, tbm.eF)
      return
   end
   # the following algorithm works for Fermi-Dirac, not general Smearing
   K, weight = monkhorstpackgrid(atm, tbm)
   Ne = tbm.norbitals * length(atm)
   nf = round(Int, ceil(Ne/2))
   # update_eig!(atm, tbm)
   # set an initial eF
   μ = 0.0
   for n = 1:size(K, 2)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      μ += weight[n] * (epsn_k[nf] + epsn_k[nf+1]) /2
   end
   # iteration by Newton algorithm
   err = 1.0
   while abs(err) > 1.0e-8
      Ni = 0.0
      gi = 0.0
      for n = 1:size(K,2)
         k = K[n]
         epsn_k = get_k_array(tbm, :epsn, k)
         Ni += weight[n] * sum_kbn( tbm.smearing(epsn_k, μ) )
         gi += weight[n] * sum_kbn( @D tbm.smearing(epsn_k, μ) )
      end
      err = Ne - Ni
      #println("\n err=");  print(err)
      μ = μ - err / gi
   end
   tbm.eF = μ
   set_eF!(tbm.smearing, tbm.eF)
end



############################################################
### Hamiltonian Evaluation


"""
`hamiltonian`: computes the hamiltonitan and overlap matrix for a tight
binding model.

#### Parameters:

* `atm::AbstractAtoms`
* `tbm::TBModel`
* `k = k=[0.;0.;0.]` : k-point at which the hamiltonian is evaluated

### Output: H, M

* `H` : hamiltoian in CSC format
* `M` : overlap matrix in CSC format
"""
function hamiltonian(atm::AbstractAtoms, tbm::TBModel, k)
   nlist = neighbourlist(atm, cutoff(tbm))
   nnz_est = length(nlist) * tbm.norbitals^2 + length(atm) * tbm.norbitals^2
   It = zeros(Int32, nnz_est)
   Jt = zeros(Int32, nnz_est)
   Ht = zeros(Complex{Float64}, nnz_est)
   Mt = zeros(Complex{Float64}, nnz_est)
   X = positions(atm)
   return hamiltonian!( tbm, k, It, Jt, Ht, Mt, nlist, X)
end

hamiltonian(tbm::TBModel, atm::AbstractAtoms) =
   hamiltonian(atm::AbstractAtoms, tbm::TBModel, JVec([0.;0.;0.]))



# TODO: the following method overload are a bit of a hack
#       it would be better to implement broadcasting in FixedSizeArrays

import Base.-
-(A::AbstractVector{JVecF}, a::JVecF) = JVecF[v - a for v in A]

dott(a::JVecF, A::AbstractVector{JVecF}) = JVecF[dot(a, v) for v in A]

function append!(It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, norbitals, idx)
   @inbounds for i = 1:norbitals, j = 1:norbitals
      idx += 1
      It[idx] = In[i]
      Jt[idx] = Im[j]
      Ht[idx] = H_nm[i,j] * exp_i_kR
      Mt[idx] = M_nm[i,j] * exp_i_kR
   end
   return idx
end

function hamiltonian!(tbm::TBModel, k, It, Jt, Ht, Mt, nlist, X)

   idx = 0                     # index into triplet format
   H_nm = zeros(tbm.norbitals, tbm.norbitals)    # temporary arrays
   M_nm = zeros(tbm.norbitals, tbm.norbitals)
   temp = zeros(10)

   # loop through sites
   for (n, neigs, r, R, _) in sites(nlist)
      In = indexblock(n, tbm)   # index-block for atom index n
      # loop through the neighbours of the current atom
      for m = 1:length(neigs)
         # note: we could use cell * S instead of R[m] - (X[neigs[m]] - X[n])
         #       but this would actually be less efficient, and less clear
         exp_i_kR = exp( im * dot(k, R[m] - (X[neigs[m]] - X[n])) )

         Im = TightBinding.indexblock(neigs[m], tbm)
         # compute hamiltonian block
         H_nm = evaluate!(tbm.hop, r[m], R[m], H_nm)
         # compute overlap block
         M_nm = evaluate!(tbm.overlap, r[m], R[m], M_nm)
         # add new indices into the sparse matrix
         idx = append!(It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, tbm.norbitals, idx)
      end
      # now compute the on-site terms
      # TODO: we could move these to be done in-place???
      # (first test: with small vectors and matrices in-place operations
      #              may become unnecessary)
      H_nn = tbm.onsite(r, R)
      M_nn = tbm.overlap(0.0)
      # add into sparse matrix
      idx = append!(It, Jt, Ht, Mt, In, In, H_nn, M_nn, 1.0, tbm.norbitals, idx)
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
   return sparse(It, Jt, Ht), sparse(It, Jt, Mt)
end



"""`densitymatrix(at::AbstractAtoms, tbm::TBModel) -> rho`:

### Input
* `at::AbstractAtoms` : configuration
* `tbm::TBModel` : calculator

### Output
* `rho::Matrix{Float64}`: density matrix,
    ρ = ∑_s f(ϵ_s) ψ_s ⊗ ψ_s
where `f` is given by `tbm.SmearingFunction`. With BZ integration, it becomes
    ρ = ∑_k w^k ∑_s f(ϵ_s^k) ψ_s^k ⊗ ψ_s^k
"""
function densitymatrix(at::AbstractAtoms, tbm::TBModel)
   update!(at, tbm)
   K, weight = monkhorstpackgrid(atm, tbm)
   rho = 0.0
   for n = 1:size(K, 2)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      C_k = get_k_array(tbm, :C, k)
      f = tbm.smearing(epsn_k, tbm.eF)
      for m = 1:length(epsn_k)
         rho += weight[n] * f[m] * C_k[:,m] * C_k[:,m]'
      end
   end
   return rho
end



############################################################
### Standard Calculator Functions


function energy(tbm::TBModel, at::AbstractAtoms)
   update!(at, tbm)
   K, weight = monkhorstpackgrid(at, tbm)
   E = 0.0
   for n = 1:size(K, 2)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      E += weight[n] * sum_kbn(tbm.smearing(epsn_k, tbm.eF) .* epsn_k)
   end
   return E
end



function band_structure_all(at::AbstractAtoms, tbm::TBModel)
   update!(at, tbm)
   na = length(at) * tbm.norbitals
   K, weight = monkhorstpackgrid(at, tbm)
   E = zeros(na, size(K,2))
   Ne = tbm.norbitals * length(at)
   nf = round(Int, ceil(Ne/2))
   for n = 1:size(K, 2)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      for j = 1:na
         E[j,n] = epsn_k[j]
      end
   end
   return K, E
end


# get 2*Nb+1 bands around the fermi level
function band_structure_near_eF(Nb, at::AbstractAtoms, tbm::TBModel)
   update!(at, tbm)
   K, weight = monkhorstpackgrid(at, tbm)
   E = zeros(2*Nb+1, size(K,2))
   Ne = tbm.norbitals * length(at)
   nf = round(Int, ceil(Ne/2))
   for n = 1:size(K, 2)
      k = K[n]
      epsn_k = get_k_array(tbm, :epsn, k)
      E[Nb+1,n] = epsn_k[nf]
      for j = 1:Nb
         E[Nb+1-j,n] = epsn_k[nf-j]
         E[Nb+1+j,n] = epsn_k[nf+j]
      end
   end
   return K, E
end

# TODO: check the S thing and probably don't pass X !!!!!
function forces_k(X::JPtsF, tbm::TBModel, nlist, k::JVecF)
   # obtain the precomputed arrays
   epsn = get_k_array(tbm, :epsn, k)
   C = get_k_array(tbm, :C, k)
   df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))

   # precompute some products
   const C_df_Ct = (C * (df' .* C)')::Matrix{Complex{Float64}}
   const C_dfepsn_Ct = (C * ((df.*epsn)' .* C)')::Matrix{Complex{Float64}}

   # allocate forces
   const frc = zeros(Complex{Float64}, 3, length(X))

   # pre-allocate dH, with a (dumb) initial guess for the size
   # TODO: re-interpret these as arrays of JVecs, where the first argument is the JVec
   const dH_nn = zeros(3, tbm.norbitals, tbm.norbitals, 6)
   const dH_nm = zeros(3, tbm.norbitals, tbm.norbitals)
   const dM_nm = zeros(3, tbm.norbitals, tbm.norbitals)

   # loop through all atoms, to compute the force on atm[n]
   for (n, neigs, r, R, _) in sites(nlist)
      # neigs::Vector{Int}   # TODO: put this back in?!?  > PROFILE IT AGAIN
      # R::Matrix{Float64}  # TODO: put this back in?!?
      #   IT LOOKS LIKE the type of R is not inferred!
      # compute the block of indices for the orbitals belonging to n
      In = indexblock(n, tbm)

      # compute ∂H_mm/∂y_n (onsite terms) M_nn = const ⇒ dM_nn = 0
      # dH_nn should be 3 x norbitals x norbitals x nneigs
      if length(neigs) > size(dH_nn, 4)
         dH_nn = zeros(3, tbm.norbitals, tbm.norbitals, ceil(Int, 1.5*length(neigs)))
      end
      evaluate_d!(tbm.onsite, r, R, dH_nn)

      for i_n = 1:length(neigs)
         m = neigs[i_n]
         Im = indexblock(m, tbm)
         tmp = R[i_n] - (X[neigs[i_n]] - X[n])
         kR = dot(tmp, k)
         eikr = exp(im * kR)::Complex{Float64}
         # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
         grad!(tbm.hop, r[i_n], - R[i_n], dH_nm)
         grad!(tbm.overlap, r[i_n], - R[i_n], dM_nm)

         # the following is a hack to put the on-site assembly into the
         # innermost loop
         # F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
         for a = 1:tbm.norbitals, b = 1:tbm.norbitals
            t1 = 2.0 * real(C_df_Ct[Im[a], In[b]] * eikr)
            t2 = 2.0 * real(C_dfepsn_Ct[Im[a], In[b]] * eikr)
            t3 = C_df_Ct[In[a],In[b]]
            # add contributions to the force
            # TODO: can re-write this as sum over JVecs
            for j = 1:3
               frc[j,n] += - dH_nm[j,a,b] * t1 + dM_nm[j,a,b] * t2 + dH_nn[j,a,b,i_n] * t3
               frc[j,m] += - t3 * dH_nn[j,a,b,i_n]
            end
         end

      end  # m in neigs-loop
   end  #  sites-loop

   # TODO: in the future assemble the forces already in JVecsF format
   return real(frc) |> vecs
end



function forces(tbm::TBModel, atm::AbstractAtoms)
   update!(atm, tbm)
   nlist = neighbourlist(atm, cutoff(tbm))
   K, weight = monkhorstpackgrid(atm, tbm)
   X = positions(atm)
   frc = zerovecs(length(atm))
   for iK = 1:size(K,2)
      frc +=  weight[iK] * forces_k(X, tbm, nlist, K[iK])
   end
   return frc
end
