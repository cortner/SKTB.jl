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
type TBModel <: AbstractTBModel
    # Hamiltonian parameters
    onsite::SitePotential
    hop::PairPotential
    overlap::PairPotential
    
    pair::PairPotential

	# HJ: add a parameter Rcut
	# since the functions "cutoff" in Potentials.jl and NRLTB.jl may conflict
	Rcut::Float64
    
    # remaining model parameters
    smearing::SmearingFunction
    norbitals::Integer

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

# HJ: not sure this returns right Rcut for NRL ----------------------------------
cutoff(tbm::TBModel) = tbm.Rcut
# import Potentials.cutoff
# max(cutoff(tbm.hop), cutoff(tbm.onsite), cutoff(tbm.pair))
# -------------------------------------------- ----------------------------------


"""`indexblock`:
a little auxiliary function to compute indices for several orbitals
"""
indexblock(n::Union{Integer, Vector}, tbm::TBModel) =
    (n-1) * tbm.norbitals .+ [1:tbm.norbitals;]'






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



"""`sorted_eig`:  helper function to compute eigenvalues, then sort them
in ascending order and sort the eig-vectors as well."""
function sorted_eig(H, M)
    #epsn, C = eig(Hermitian(full(H)), Hermitian(full(M)))
    HH = full(H)
    MM = full(M)
    HS = (HH + HH') / 2.0
    MS = (MM + MM') / 2.0
    epsn, C = eig(HS, MS)   
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
    if kx != 0 || ky != 0
	error("This boundary condition has not been implemented yet!")
    end
    # open boundarycondition OR Γ-point sampling
    if kz == 0 || kz == 1
	K = [0.;0.;0.]
	weight = 1.0
    else
	if mod(kz,2) == 1
	    error("k should be an even number in Monkhorst-Pack grid!")
	end
   	# compute the lattice vector of reciprocal space
	v1 = cell[1,:][:]
    	v2 = cell[2,:][:]
    	v3 = cell[3,:][:]
    	c12 = cross(v1,v2)
    	b3 = 2 * π * c12 / dot(v3,c12)
	## MonkhorstPack: K = {b/kz * j + shift}_{j=-kz/2+1,...,kz/2} with shift = 0.0
	#  We can exploit the symmetry of the brillouin zone 
	nk = Int(kz/2) + 1
	K = zeros(nk, 3)
	weight = zeros(nk)
	k_step = b3 / kz
    	w_step = norm(b3) / kz+1
    	for k = 1:nk
            K[k,:] = (k-1) * k_step
	    if k == 1 || k == nk
		weight[k] = w_step
	    else 
	        weight[k] = w_step * 2.0 
	    end
    	end
    end
    return K, weight
end



"""`monkhorstpackgrid(atm::ASEAtoms, tbm::TBModel)` : extracts cell and grid 
    information and returns an MP grid.
"""
monkhorstpackgrid(atm::ASEAtoms, tbm::TBModel) =
    monkhorstpackgrid(cell(atm), tbm.nkpoints)



############################################################
##### update functions

"""`update_eig!(atm::ASEAtoms, tbm::TBModel, k)` : computes hamiltonian for
 k-point `k`, diagonalises and stores the  diagonalisation in `tbm.arrays`
"""
function update_eig!(atm::ASEAtoms, tbm::TBModel, k)
    H, M = hamiltonian(atm, tbm, k)
    epsn, C = sorted_eig(H, M)
    set_k_array!(tbm, epsn, :epsn, k)
    set_k_array!(tbm, C, :C, k)
end


"""`update_eig!(atm::ASEAtoms, tbm::TBModel)` : updates the hamiltonians
and spectral decompositions on the MP grid.
"""
function update_eig!(atm::ASEAtoms, tbm::TBModel)
    K, weight = monkhorstpackgrid(atm, tbm)
    for n = 1:size(K, 2)
        update_eig!(atm, tbm, K[:, n])
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
	nf = ceil( Ne / 2 ) 
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
			Ni += weight[n] * r_sum( fermi_dirac(μ, tbm.beta, epsn_k) )
			gi += weight[n] * r_sum( fermi_dirac_d(μ, tbm.beta, epsn_k) )
		end
	    μ = μ - err / gi
	    err = Ne - Ni
	    # println(err)
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

    # create a neighbourlist
    nlist = NeighbourList(cutoff(tbm), atm)
    # setup a huge sparse matrix, we need a rough estimate for the number of
    # >> ask nlist how much storage we roughly need!
    nnz_est = 2 * length(nlist.Q['i']) * tbm.norbitals^2 
    # allocate space for hamiltonian and overlap matrix
    H = sparse_flexible(nnz_est, Complex{Float64})
    M = sparse_flexible(nnz_est, Complex{Float64})
    
    X = positions(atm)
    # loop through all atoms
    for (n, neigs, r, R) in Sites(nlist)
        # index-block for atom index n
        In = indexblock(n, tbm)
        # loop through the neighbours of the current atom
        for m = 1:length(neigs)
            # get the block of indices for atom m
            Im = indexblock(neigs[m], tbm)
            kR = dot(R[:,m] - (X[:,neigs[m]] - X[:,n]), k)
            # compute hamiltonian block and add to sparse matrix
            H_nm = tbm.hop(r[m], R[:, m])        # OLD: get_h!(R[:,m], tbm, H_nm)
            H[In, Im] += H_nm * exp(im * kR)
            # compute overlap block and add to sparse matrix
            M_nm = tbm.overlap(r[m], R[:,m])     # OLD: get_m!(R[:.m], tbm, M_nm)
            M[In, Im] += M_nm * exp(im * kR)   
        end
        # now compute the on-site terms
        H_nn = tbm.onsite(r, R)                  # OLD: get_os!(R, tbm, H_nm)
        H[In, In] += H_nn
        # overlap diagonal block
        M_nn = tbm.overlap(0.0)
        M[In, In] += M_nn
    end
    # convert M, H and return
    return sparse_static(H), sparse_static(M)
end


hamiltonian(atm::ASEAtoms, tbm::TBModel) =
    hamiltonian(atm::ASEAtoms, tbm::TBModel, [0.;0.;0.])
                     

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




# compute all forces on all the atoms
function forces(atm::ASEAtoms, tbm::TBModel)
    # tell tbm to update the spectral decompositions
    update!(atm, tbm)
    
    # allocate output
	dim = 3
    Natm = length(atm)
    frc = zeros(Complex{Float64}, dim, Natm)

    # precompute neighbourlist
    nlist = NeighbourList(cutoff(tbm), atm)

    # BZ integration loop
    K, weight = monkhorstpackgrid(atm, tbm)
    for iK = 1:size(K,2)
        k = K[:, iK]
        epsn = get_k_array(tbm, :epsn, k)
        C = get_k_array(tbm, :C, k)
        df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))
        ##### TODO: HOW DOES SMEARING KNOW eF ????
    
    	X = positions(atm)
        # loop through all atoms, to compute the force on atm[n]
        for (n, neigs, r, R) in Sites(nlist)
            # compute the block of indices for the orbitals belonging to n
            In = indexblock(n, tbm)

            # compute ∂H_mm/∂y_n (onsite terms) M_nn = const ⇒ dM_nn = 0
            # dH_nn should be 3 x norbitals x norbitals x nneigs
            dH_nn = @D tbm.onsite(r, R)
            # IN THE NEW FRAMEWORK THIS SHOULD RETURN A 3-DIMENSIONAL
            # ARRAY WITH ALL THE DERIVATIVES W.R.T. ALL THE SITES!!!
            
            # HOPPING TERMS
            # loop through neighbours of atm[n]
            for i_n = 1:length(neigs)
             	kR = dot(R[:,i_n] - (X[:,neigs[i_n]] - X[:,n]), k)
				eikr = exp(im * kR)

                m = neigs[i_n]
	            Im = indexblock(m, tbm)
                # compute ∂H_nm/∂y_n (hopping terms) and ∂M_nm/∂y_n
                dH_nm = @GRAD tbm.hop(r[i_n], -R[:, i_n])
                dM_nm =  @GRAD tbm.overlap(r[i_n], -R[:,i_n])
                
   				# NOTE: there is still DERIVATIVE WITH RESPECT TO exp(ikR)
                # compute hamiltonian block and add to sparse matrix
    	        #H_nm = tbm.hop(r[i_n], R[:, i_n])
    	        #H_nm = tbm.hop(r[i_n], R[:, i_n]) * exp(0.0*im) 
            	# compute overlap block and add to sparse matrix
	            #M_nm = tbm.overlap(r[i_n], R[:,i_n])
	            #M_nm = tbm.overlap(r[i_n], R[:,i_n]) * exp(0.0*im)    
				# H = exp(ikR) * h  ⇒  dH = exp(ikR) * (dh + ik*h)
				#for d = 1:dim
				#	# NOT SURE whether the following lines are well written ...
				#	dH_nm[d,:][:] = eikr * ( dH_nm[d,:][:] + im * k[d] * H_nm[:] )
				#	dM_nm[d,:][:] = eikr * ( dM_nm[d,:][:] + im * k[d] * M_nm[:] )
				#end
                
                # the following is a hack to put the on-site assembly into the
                # innermost loop                
                # F_n = - ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
                for a = 1:tbm.norbitals, b = 1:tbm.norbitals
                    t1 = 0.0; t2 = 0.0; t3 = 0.0 
                    ina = In[a]; ima = Im[a]; inb = In[b]; imb = Im[b]
                    @inbounds @simd for s = 1:length(epsn)
                        t1 += df[s] * C[ima,s] * C[inb,s]'
                        t2 += df[s] * C[ima,s] * C[inb,s]' * epsn[s]
                        t3 += df[s] * C[ina,s] * C[inb,s]'
                    end
                    # add contributions to the force
                    frc[:,n] -= weight[iK] * ( dH_nm[:,a,b] * (t1 * eikr + t1' * eikr') 
                                  - dM_nm[:,a,b] * (t2 * eikr + t2' * eikr') 
                                  # 2.0 * t1 * dH_nm[:,a,b] - 2.0 * t2 * dM_nm[:,a,b] 
                                  - t3 * dH_nn[:,a,b,i_n] )
                    frc[:,m] -= weight[iK] * t3 * dH_nn[:,a,b,i_n] 
                end
                
            end  # m in neigs-loop
        end  #  sites-loop
    end # k-loop
    
    return real(frc)
end



potential_energy_d(atm::ASEAtoms, tbm::TBModel) = -forces(atm, tbm)



############################################################
### Site Energy Stuff




##################################################
### MODELS




include("NRLTB.jl")
include("tbtoymodel.jl")

end
