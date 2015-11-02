"""
`module TightBinding`

### Summary

Implements some functionality for tight binding models

### Notes

* at the moment we assume availability of ASE, MatSciPy
"""
module TightBinding

using AtomsInterface
importall AtomsInterface

using Potentials, ASE, MatSciPy, Prototypes, SparseTools
import MatSciPy.potential_energy
import Potentials.evaluate, Potentials.evaluate_d

export AbstractTBModel, SimpleFunction
export TBModel, FermiDiracSmearing, potential_energy, forces, evaluate,
potential_energy_d


abstract AbstractTBModel <: AbstractCalculator
abstract SmearingFunction <: SimpleFunction




"""`TBModel`: basic non-self consistent tight binding calculator. This
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
    Rcut::Float64

    # remaining model parameters
    smearing::SmearingFunction
    norbitals::Int

    #  WHERE DOES THIS GO?
    fixed_eF::Bool
    eF::Float64
    # beta::Float64
    # eF::Float64

    # k-point sampling information:
    #    0 = open boundary
    #    1 = Gamma point
    nkpoints::Tuple{Int, Int, Int}

    # internals
    hfd::Float64           # step used for finite-difference approximations
    needupdate::Bool       # tells whether hamiltonian and spectrum are up-to-date
    arrays::Dict{Any, Any}      # storage for various
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


############################################################
#### UTILITY FUNCTIONS

# import Potentials.cutoff
cutoff(tbm::TBModel) = tbm.Rcut
# HJ: not sure this returns right Rcut for NRL ----------------------------------
# max(cutoff(tbm.hop), cutoff(tbm.onsite), cutoff(tbm.pair))
# -------------------------------------------- ----------------------------------


"""`indexblock`:
a little auxiliary function to compute indices for several orbitals
"""
# indexblock(n::Vector, tbm::TBModel) =
#     (n-1) * tbm.norbitals .+ [1:tbm.norbitals;]'
indexblock(n::Integer, tbm::TBModel) =
    Int[(n-1) * tbm.norbitals + j for j = 1:tbm.norbitals]



"""`FermiDiracSmearing`:

f(e) = ( 1 + exp( beta (e - eF) ) )^{-1}
"""
type FermiDiracSmearing <: SmearingFunction
    beta
    eF
end
FermiDiracSmearing(beta;eF=0.0) = FermiDiracSmearing(beta, eF)
# FD distribution and its derivative. both are vectorised implementations
fermi_dirac(eF, beta, epsn) =
    2.0 ./ ( 1.0 + exp((epsn - eF) * beta) )
fermi_dirac_d(eF, beta, epsn) =
    - fermi_dirac(eF, beta, epsn).^2 .* exp((epsn - eF) * beta) * beta / 2.0
# Boilerplate to work with the FermiDiracSmearing type
@inline evaluate(fd::FermiDiracSmearing, epsn) =
    fermi_dirac(fd.eF, fd.beta, epsn)
@inline evaluate_d(fd::FermiDiracSmearing, epsn) =
    fermi_dirac_d(fd.eF, fd.beta, epsn)
# Boilerplate to work with the FermiDiracSmearing type
@inline evaluate(fd::FermiDiracSmearing, epsn, eF) =
    fermi_dirac(eF, fd.beta, epsn)
@inline evaluate_d(fd::FermiDiracSmearing, epsn, eF) =
    fermi_dirac_d(eF, fd.beta, epsn)

function set_eF!(fd::FermiDiracSmearing, eF)
    fd.eF = eF
end



"""`ZeroTemperature`:

TODO
"""
type ZeroTemperature <: SmearingFunction
    eF
end


# TODO: need a function that determines the Fermi Level!


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

""" store k-point dependent arrays"""
set_k_array!(tbm, q, symbol, k) =  set_array!(tbm, (symbol, k), q)
"""retrieve k-point dependent arrays"""
get_k_array(tbm, symbol, k) = get_array(tbm, (symbol, k))


 """`monkhorstpackgrid(cell, nkpoints)` : constructs an MP grid for the
computational cell defined by `cell` and `nkpoints`. Returns

### Parameters

* 'cell' : 3 × 1 array of lattice vector for (super)cell
* 'nkpoints' : 3 × 1 array of number of k-points in each direction. Now
it can only be (0, 0, kz::Int).

### Output

* `K`: 3 × Nk array of k-points
* `weights`: integration weights; scalar (uniform grid) or Nk array.
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
	v1 = cell[1,:][:]
    v2 = cell[2,:][:]
    v3 = cell[3,:][:]
    c12 = cross(v1,v2)
	c23 = cross(v2,v3)
	c31 = cross(v3,v1)
	b1 = 2 * π * c23 / dot(v1,c23)
	b2 = 2 * π * c31 / dot(v2,c31)
	b3 = 2 * π * c12 / dot(v3,c12)
	## MonkhorstPack: K = {b/kz * j + shift}_{j=-kz/2+1,...,kz/2}
	# with shift = 0.0.
	# We can exploit the symmetry of the BZ.
	## NOTE THAT this is not necessarily first BZ and
    # THE SYMMETRY HAS NOT BEEN FULLY EXPLOITED YET!!
	nx = Int(kx/2) + 1
	ny = Int(ky/2) + 1
	nz = Int(kz/2) + 1
	N = nx * ny * nz
	K = zeros(3, N)
	weight = zeros(N)

	kx_step = b1 / (kx==0? 1:kx)
	ky_step = b2 / (ky==0? 1:ky)
	kz_step = b3 / (kz==0? 1:kz)
    w_step = 1.0 / ( (kx==0? 1:kx) * (ky==0? 1:ky) * (kz==0? 1:kz) )
	# evaluate K and weight 
   	for k1 = 1:nx, k2 = 1:ny, k3 = 1:nz
		k = k1 + (k2-1) * nx + (k3-1) * nx * ny
        K[:,k] = (k1-1) * kx_step + (k2-1) * ky_step + (k3-1) * kz_step
		# adjust weight by symmetry
		weight[k] = w_step * 8.0
    	if k1 == 1 || k1 == nx
			weight[k] = weight[k] / 2.0
		end
      	if k2 == 1 || k2 == ny
			weight[k] = weight[k] / 2.0
		end
    	if k3 == 1 || k3 == nz
			weight[k] = weight[k] / 2.0
		end
    end
	#print(K); println("\n"); print(weight); println("\n")
	#println("sum_weight = "); print(sum(weight))
    return K, weight
end



"""`monkhorstpackgrid(atm::ASEAtoms, tbm::TBModel)` : extracts cell and grid
    information and returns an MP grid.
"""
monkhorstpackgrid(atm::ASEAtoms, tbm::TBModel) =
    monkhorstpackgrid(cell(atm), tbm.nkpoints)



############################################################
##### update functions



"""`update_eig!(atm::ASEAtoms, tbm::TBModel)` : updates the hamiltonians
and spectral decompositions on the MP grid.
"""
function update_eig!(atm::ASEAtoms, tbm::TBModel)
    K, weight = monkhorstpackgrid(atm, tbm)
    nlist = NeighbourList(cutoff(tbm), atm)
    nnz_est = length(nlist) * tbm.norbitals^2 + length(atm) * tbm.norbitals^2
    It = zeros(Int32, nnz_est)
    Jt = zeros(Int32, nnz_est)
    Ht = zeros(Complex{Float64}, nnz_est)
    Mt = zeros(Complex{Float64}, nnz_est)
    X = positions(atm)
    for n = 1:size(K, 2)
        k = K[:,n]
        H, M = hamiltonian!(tbm, k, It, Jt, Ht, Mt, nlist, X)
        epsn, C = sorted_eig(H, M)
        set_k_array!(tbm, epsn, :epsn, k)
        set_k_array!(tbm, C, :C, k)
    end
end


"""`update!(atm::ASEAtoms, tbm:TBModel)`: checks whether the precomputed
data stored in `tbm` needs to be updated (by comparing atom positions) and
if so, does all necessary updates. At the moment, the following are updated:

* spectral decompositions (`update_eig!`)
* the fermi-level (`update_eF!`)
"""
function update!(atm::ASEAtoms, tbm::TBModel)
    Xnew = positions(atm)
    Xold = tbm[:X]   # (returns nothing if X has not been stored previously)
    if Xnew != Xold
        tbm[:X] = Xnew
        # do all the updates
        update_eig!(atm, tbm)
        update_eF!(atm, tbm)
    end
end


"""`update_eF!(tbm::TBModel)`: recompute the correct
fermi-level; using the precomputed data in `tbm.arrays`
"""
function update_eF!(atm::ASEAtoms, tbm::TBModel)
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
        k = K[:, n]
        epsn_k = get_k_array(tbm, :epsn, k)
		μ += weight[n] * (epsn_k[nf] + epsn_k[nf+1]) /2
    end
	# iteration by Newton algorithm
	err = 1.0
	while abs(err) > 1.0e-8
		Ni = 0.0
		gi = 0.0
		for n = 1:size(K,2)
	        k = K[:, n]
    	    epsn_k = get_k_array(tbm, :epsn, k)
			Ni += weight[n] * r_sum( tbm.smearing(epsn_k, μ) )
			gi += weight[n] * r_sum( @D tbm.smearing(epsn_k, μ) )
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


"""`hamiltonian`: computes the hamiltonitan and overlap matrix for a tight
binding model.

#### Parameters:

* `atm::ASEAtoms`
* `tbm::TBModel`
* `k = k=[0.;0.;0.]` : k-point at which the hamiltonian is evaluated

### Output: H, M

* `H` : hamiltoian in CSC format
* `M` : overlap matrix in CSC format
    """
function hamiltonian(atm::ASEAtoms, tbm::TBModel, k)
    nlist = NeighbourList(cutoff(tbm), atm)
    nnz_est = length(nlist) * tbm.norbitals^2 + length(atm) * tbm.norbitals^2
    It = zeros(Int32, nnz_est)
    Jt = zeros(Int32, nnz_est)
    Ht = zeros(Complex{Float64}, nnz_est)
    Mt = zeros(Complex{Float64}, nnz_est)
    X = positions(atm)
    return hamiltonian!( tbm, k, It, Jt, Ht, Mt, nlist, X)
end

hamiltonian(atm::ASEAtoms, tbm::TBModel) =
    hamiltonian(atm::ASEAtoms, tbm::TBModel, [0.;0.;0.])


function hamiltonian!(tbm::TBModel, k,
                      It, Jt, Ht, Mt, nlist, X)
    
    idx = 0                     # index ito triplet format
    H_nm = zeros(tbm.norbitals, tbm.norbitals) # temporary arrays
    M_nm = zeros(tbm.norbitals, tbm.norbitals)
    temp = zeros(10)
    
    # loop through sites
    for (n, neigs, r, R, _) in Sites(nlist)
        In = indexblock(n, tbm)   # index-block for atom index n
        exp_i_kR = exp(im * (k' * (R - (X[:,neigs] .- X[:,n]))))
        # loop through the neighbours of the current atom
        for m = 1:length(neigs)
            Im = TightBinding.indexblock(neigs[m], tbm)
            # compute hamiltonian block
            H_nm = evaluate!(tbm.hop, r[m], R[:, m], H_nm)
            # compute overlap block
            M_nm = evaluate!(tbm.overlap, r[m], R[:, m], M_nm)
            # add new indices into the sparse matrix
            @inbounds for i = 1:tbm.norbitals, j = 1:tbm.norbitals
                idx += 1
                It[idx] = In[i]
                Jt[idx] = Im[j]
                Ht[idx] = H_nm[i,j]*exp_i_kR[m]
                Mt[idx] = M_nm[i,j]*exp_i_kR[m]
            end
        end
        # now compute the on-site terms (we could move these to be done in-place)
        H_nn = tbm.onsite(r, R)
        M_nn = tbm.overlap(0.0)
        # add into sparse matrix
        for i = 1:tbm.norbitals, j = 1:tbm.norbitals
            idx += 1
            It[idx] = In[i]
            Jt[idx] = In[j]
            Ht[idx] = H_nn[i,j]
            Mt[idx] = M_nn[i,j]
        end
    end
    
    # convert M, H into Sparse CCS and return
    #   TODO: The conversion to sparse format accounts for about 1/2 of the
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



"""`densitymatrix(at::ASEAtoms, tbm::TBModel) -> rho`:

### Input
* `at::ASEAtoms` : configuration
* `tbm::TBModel` : calculator

### Output
* `rho::Matrix{Float64}`: density matrix,
    ρ = ∑_s f(ϵ_s) ψ_s ⊗ ψ_s
where `f` is given by `tbm.SmearingFunction`. With BZ integration, it becomes
    ρ = ∑_k w^k ∑_s f(ϵ_s^k) ψ_s^k ⊗ ψ_s^k
"""
function densitymatrix(at::ASEAtoms, tbm::TBModel)
    update!(at, tbm)
    K, weight = monkhorstpackgrid(atm, tbm)
    rho = 0.0
    for n = 1:size(K, 2)
        k = K[:, n]
        epsn_k = get_k_array(tbm, :epsn, k)
        C_k = get_k_array(tbm, :C, k)
        f = tbm.smearing(epsn_k, tbm.eF)
        # TODO: should eF be passed or should it be sotred in SmearingFunction?
        for m = 1:length(epsn_k)
            rho += weight[n] * f[m] * C_k[:,m] * C_k[:,m]'
        end
    end
    return rho
end



############################################################
### Standard Calculator Functions


function potential_energy(at::ASEAtoms, tbm::TBModel)

    update!(at, tbm)

    K, weight = monkhorstpackgrid(at, tbm)
    E = 0.0
    for n = 1:size(K, 2)
        k = K[:, n]
        epsn_k = get_k_array(tbm, :epsn, k)
        E += weight[n] * r_sum(tbm.smearing(epsn_k, tbm.eF) .* epsn_k)
        # TODO: pass eF?
    end

    return E
end



function forces_k(X::Matrix{Float64}, tbm::TBModel, nlist, k::Vector{Float64})

    # obtain the precomputed arrays
    epsn = get_k_array(tbm, :epsn, k)
    C = get_k_array(tbm, :C, k)
    df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))

    # precompute some products
    const C_df_Ct = (C * (df' .* C)')::Matrix{Complex{Float64}}
    const C_dfepsn_Ct = (C * ((df.*epsn)' .* C)')::Matrix{Complex{Float64}}

    # allocate forces
    const frc = zeros(Complex{Float64}, 3, size(X,2))

    # pre-allocate dH, with a (dumb) initial guess for the size
    const dH_nn = zeros(3, tbm.norbitals, tbm.norbitals, 6)
    const dH_nm = zeros(3, tbm.norbitals, tbm.norbitals)
    const dM_nm = zeros(3, tbm.norbitals, tbm.norbitals)
    
    # loop through all atoms, to compute the force on atm[n]
    for (n, neigs, r, R) in Sites(nlist)
        neigs::Vector{Int}
        R::Matrix{Float64}
        # compute the block of indices for the orbitals belonging to n
        In = indexblock(n, tbm)

        # compute ∂H_mm/∂y_n (onsite terms) M_nn = const ⇒ dM_nn = 0
        # dH_nn should be 3 x norbitals x norbitals x nneigs
        # dH_nn = (@D tbm.onsite(r, R))::Array{Float64,4}
        # in-place version
        if length(neigs) > size(dH_nn, 4)
            dH_nn = zeros(3, tbm.norbitals, tbm.norbitals,
                          ceil(Int, 1.5*length(neigs)))
        end
        evaluate_d!(tbm.onsite, r, R, dH_nn)

        for i_n = 1:length(neigs)
            m = neigs[i_n]
	    Im = indexblock(m, tbm)
            kR = dot(R[:,i_n] - (X[:,neigs[i_n]] - X[:,n]), k)
	    eikr = exp(im * kR)::Complex{Float64}
            # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
            # dH_nm = (@GRAD tbm.hop(r[i_n], -R[:, i_n]))::Array{Float64,3}
            # dM_nm = (@GRAD tbm.overlap(r[i_n], -R[:,i_n]))::Array{Float64,3}
            grad!(tbm.hop, r[i_n], -R[:,i_n], dH_nm)
            grad!(tbm.overlap, r[i_n], -R[:,i_n], dM_nm)
            
            # the following is a hack to put the on-site assembly into the
            # innermost loop
            # F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
            for a = 1:tbm.norbitals, b = 1:tbm.norbitals
                t1 = 2.0 * real(C_df_Ct[Im[a], In[b]] * eikr)
                t2 = 2.0 * real(C_dfepsn_Ct[Im[a], In[b]] * eikr)
                t3 = C_df_Ct[In[a],In[b]]
                # add contributions to the force
                for j = 1:3
                    frc[j,n] = frc[j,n] - dH_nm[j,a,b] * t1 +
                                       dM_nm[j,a,b] * t2 + dH_nn[j,a,b,i_n] * t3
                    frc[j,m] = frc[j,m] - t3 * dH_nn[j,a,b,i_n]
                end
            end

        end  # m in neigs-loop
    end  #  sites-loop

    return frc

end


function forces(atm::ASEAtoms, tbm::TBModel)
    # tell tbm to update the spectral decompositions
    update!(atm, tbm)
    # allocate output
    frc = zeros(3, length(atm))

    # precompute neighbourlist
    nlist = NeighbourList(cutoff(tbm), atm)
    X = positions(atm)
    # BZ integration loop
    K, weight = monkhorstpackgrid(atm, tbm)
    ctr = 0
    for iK = 1:size(K,2)
        ctr += 2 * size(get_k_array(tbm, :C, K[:,iK]), 1)^2
        frc +=  weight[iK] * real(forces_k(X, tbm, nlist, K[:,iK]))
    end
    @show ctr
    return frc
end


# pull the k-loop inside the site-loop
function forces_(atm::ASEAtoms, tbm::TBModel)
    
    # tell tbm to update the spectral decompositions
    update!(atm, tbm)
    # allocate output
    const frc = zeros(3, length(atm))
    
    # precompute neighbourlist
    nlist = NeighbourList(cutoff(tbm), atm)
    X = positions(atm)::Matrix{Float64}
    K, weight = monkhorstpackgrid(atm, tbm)    
    
    # obtain the precomputed arrays
    df = zeros(length(atm)*tbm.norbitals, size(K,2))::Matrix{Float64}
    dfe = zeros(length(atm)*tbm.norbitals, size(K,2))::Matrix{Float64}
    for iK = 1:size(K,2)
     	epsn = get_k_array(tbm, :epsn, K[:,iK])::Vector{Float64}
     	df[:,iK] = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))
     	dfe[:,iK] = df[:,iK] .* epsn
    end
    df::Matrix{Float64}
    dfe::Matrix{Float64}
    
    ctr = 0
    
    # pre-allocate dH, with a (dumb) initial guess for the size
    const dH_nn = zeros(3, tbm.norbitals, tbm.norbitals, 6)
    const dH_nm = zeros(3, tbm.norbitals, tbm.norbitals)
    const dM_nm = zeros(3, tbm.norbitals, tbm.norbitals)

    for (n, neigs, r, R, _) in Sites(nlist)
        # compute the block of indices for the orbitals belonging to n
        In = indexblock(n, tbm)

        # compute ∂H_mm/∂y_n (onsite terms) M_nn = const ⇒ dM_nn = 0
        # dH_nn should be 3 x norbitals x norbitals x nneigs
        # dH_nn = (@D tbm.onsite(r, R))::Array{Float64,4}
        # in-place version
        if length(neigs) > size(dH_nn, 4)
            dH_nn = zeros(3, tbm.norbitals, tbm.norbitals,
                          ceil(Int, 1.5*length(neigs)))
        end
        evaluate_d!(tbm.onsite, r, R, dH_nn)

        for i_n = 1:length(neigs)
     	    m = neigs[i_n]
	    Im = indexblock(m, tbm)
            # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
            # dH_nm = (@GRAD tbm.hop(r[i_n], -R[:, i_n]))::Array{Float64,3}
            # dM_nm = (@GRAD tbm.overlap(r[i_n], -R[:,i_n]))::Array{Float64,3}
            grad!(tbm.hop, r[i_n], -R[:,i_n], dH_nm)
            grad!(tbm.overlap, r[i_n], -R[:,i_n], dM_nm)
            
  	    for iK = 1:size(K,2)
		k = K[:,iK]
	        kR = dot(R[:,i_n] - (X[:,neigs[i_n]] - X[:,n]), k)
		eikr = exp(im * kR) # ::Complex{Float64}
                
		C = get_k_array(tbm, :C, k)::Matrix{Complex{Float64}}
		# C_df_Ct = (C * (df[:,iK]' .* C)')
		# C_dfepsn_Ct = (C * (dfe[:,iK]' .* C)')
                
          	# the following is a hack to put the on-site assembly into the
	        # innermost loop
    	        # F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
        	for a = 1:tbm.norbitals, b = 1:tbm.norbitals
                    Ima = Im[a]; Ina = In[a]; Inb = In[b]
                    t1 = t2 = t3 = im*0.0
                    ctr += 3
                    @inbounds @simd for s = 1:size(C,2)
                        t1 += df[s,iK] * C[Ima,s] * C[Inb,s]' # C_df_Ct[Im[a], In[b]]
                        t2 += dfe[s,iK] * C[Ima,s] * C[Inb,s]' # C_dfepsn_Ct[Im[a], In[b]]
                        t3 += df[s,iK] * C[Ina,s] * C[Inb,s]' # C_df_Ct[In[a],In[b]]
                    end
                    s1 = 2.0 * real(t1 * eikr)
                    s2 = 2.0 * real(t2 * eikr)
                    s3 = real(t3)
            	    # t1 = 2.0 * real(C_df_Ct[Im[a], In[b]] * eikr)
                    # t2 = 2.0 * real(C_dfepsn_Ct[Im[a], In[b]] * eikr)
	            # t3 = real(C_df_Ct[In[a],In[b]])
    	            # add contributions to the force
                    @inbounds for j = 1:3
                        frc[j,n] = frc[j,n] - weight[iK] * ( dH_nm[j,a,b] * s1 -
                                    dM_nm[j,a,b] * s2 - dH_nn[j,a,b,i_n] * s3 )
                        frc[j,m] = frc[j,m] - weight[iK] * s3 * dH_nn[j,a,b,i_n]
                    end
                end
                
	    end  # BZ integal k-loop
       	end  # m in neigs-loop
    end  # sites-loop

    @show ctr
    
    return frc
end




# compute all forces on all the atoms
function forces_debug(atm::ASEAtoms, tbm)
    # tell tbm to update the spectral decompositions
    update!(atm, tbm)
    # allocate output
    frc = zeros(3, length(atm))
    # precompute neighbourlist
    nlist = NeighbourList(cutoff(tbm), atm)
    X = positions(atm)

    @code_warntype forces_k(X, tbm, nlist, zeros(3))
end




potential_energy_d(atm::ASEAtoms, tbm::TBModel) = -forces(atm, tbm)



############################################################
### Site Energy Stuff




##################################################
### MODELS




include("NRLTB.jl")
include("tbtoymodel.jl")

end
