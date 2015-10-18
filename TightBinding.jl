"""
`module TightBinding`

### Summary

Implements some functionality for tight binding models

### Notes

* at the moment we assume availability of ASE, MatSciPy
"""
module TightBinding




using Potentials, ASE, MatSciPy, Prototypes, SparseTool

export AbstractTBModel, TBModel

abstract AbstractTBModel <: AbstractCalculator






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
        smearing = ZeroTemperature(),
        norbitals = 0,
        fixed_eF = true,
        eF = 0.0,
        nkpoints = (0,0,0),
        hfd = 0.0,
        needupdate = true,
        arrays = Dict()) =
            TBModel(onsite, hop, overlap, pair, smearing, norbitals,
                    fixed_eF, eF, nkpoints, hfd, needupdate, arrays)


############################################################
#### UTILITY FUNCTIONS

cutoff(tbm::TBModel) = max(cutoff(tbm.hopping), cutoff(tbm.onsite),
                           cutoff(tbm.pair))


"""`indexblock`:
a little auxiliary function to compute indices for several orbitals
"""
indexblock(n::Union{Integer, Vector}, tbm::TBModel) =
    (n-1) * tbm.norbitals .+ [1:tbm.norbitals;]'




abstract SmearingFunction <: SimpleFunction


"""`FermiDiracSmearing`: 

f(e) = ( 1 + exp( beta (e - eF) ) )^{-1}
"""
type FermiDiracSmearing <: SmearingFunction
    beta
end
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


"""`ZeroTemperature`: 

TODO
"""
type ZeroTemperature <: SmearingFunction
    eF
end


# TODO: need a function that determines the Fermi Level!



"""`sorted_eig`:  helper function to compute eigenvalues, then sort them
in ascending order and sort the eig-vectors as well."""->
function sorted_eig(H, M)
    epsn, C = eig(Hermitian(full(H)), Hermitian(full(M)))
    Isort = sortperm(epsn)
    return epsn[Isort], C[:, Isort]
end


""" store k-point dependent arrays"""
function set_k_array!(tbm, q, symbol, k)
    tbm.arrays[(symbol, k)] = q
end

"""retrieve k-point dependent arrays"""
function get_k_array(tbm, symbol, k)
    tbm.arrays[(symbol, k)]
end


"""`monkhorstpackgrid(atm::ASEAtoms, tbm::TBModel)` : extracts cell and grid 
    information and returns an MP grid.
"""
monkhorstpackgrid(atm::ASEAtoms, tbm::TBModel) =
    monkhorstpackgrid(cell(atm), tbm.nkpoints)

# HJ ------------------------------------------------------------------------
"""`monkhorstpackgrid(cell, nkpoints)` : constructs an MP grid for the 
computational cell defined by `cell` and `nkpoints`. Returns 

### Parameters

(TODO HUAJIE)

### Returns

* `K`: 3 × Nk array of k-points
* `weights`: integration weights; scalar (uniform grid) or Nk array.
"""
function monkhorstpackgrid(cell::Array, nkpoints::Vector)
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
	# K = {b/kz * j + shift}_{j=-kz/2+1,...,kz/2} with shift = 0.0
	# so we can exploit the symmetry of the brillouin zone 
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
# ------------------------------------------------------------------------


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
function update!(atm::ASEAtoms, tbm:TBModel)
    Xnew = get_positions(atm)
    if haskey(tbm.arrays, :X)
        Xold = get_array(tbm, :X)
    else
        Xold = nothing
    end
    if Xnew != Xold
        tbm[:X] = Xnew
        # do all the updates
        update_eig!(atm, tbm)
        update_eF!(tbm)
    end
end


"""`update_eF!(tbm::TBModel)`: recompute the correct
fermi-level; using the precomputed data in `tbm.arrays`
"""
function update_eF!(tbm::TBModel)
    if tbm.fixed_eF
        return
    end
    
    # (TODO HUAJIE)
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

"""->
function hamiltonian(atm::ASEAtoms, tbm::TBModel; k=[0.;0.;0.])

    # create a neighbourlist
    nlist = NeighbourList(rcut(tbm), atm)
    # setup a huge sparse matrix, we need a rough estimate for the number of
    # >> ask nlist how much storage we roughly need!
    nnz_est = 2 * length(nlist.Q['i']) * tbm.norbitals^2 
    # allocate space for hamiltonian and overlap matrix
    H = sparse_flexible(nnz_est)
    M = sparse_flexible(nnz_est)
    # OLD: H_nm = zeros(tbm.norbitals, tbm.norbitals)
    # OLD: M_nm = zeros(tbm.norbitals, tbm.norbitals)
    
	X = get_positions(atm)
    # loop through all atoms
    for n, neigs, r, R in Sites(nlist)
        # index-block for atom index n
        In = indexblock(n, tbm)
        # loop through the neighbours of the current atom
        for m = 1:length(neigs)
            # get the block of indices for atom m
            Im = indexblock(neigs[m], tbm)
            # compute hamiltonian block and add to sparse matrix
# HJ----------------------------------------------------------------------------
            # DISCUSS: ARE WE USING MINIMAL IMAGE CONVENTION HERE? (OR ANYWHERE?) I DONT THINK WE CAN!!!
            kR = dot(R[:,m] - (X[:,neigs[m]] - X[:,n]), k)
            H_nm = tbm.hop(r[m], R[:, m])        # OLD: get_h!(R[:,m], tbm, H_nm)
            H[In, Im] += H_nm * exp(im * kR)
            # compute overlap block and add to sparse matrix
            M_nm = tbm.overlap(r[m], R[:,m])     # OLD: get_m!(R[:.m], tbm, M_nm)
            M[In, Im] += M_nm * exp(im * kR)   
# ------------------------------------------------------------------------------
        end
        # now compute the on-site terms
        H_nn = tbm.onsite(r, R)               #  OLD: get_os!(R, tbm, H_nm)
        H[In, In] += H_nn
        # overlap diagonal block
        M_nn = tbm.overlap(0.0)
        M[In, In] += M_nn
    end
    
    # convert M, H and return
    return sparse_static(H), sparse_static(M)
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
        for m = 1:length(epsn_k)
            rho += weight[n] * f[m] * C_k[:,m]*C_k[:,m]'
        end
    end
    return rho
end


############################################################
### Standard Calculator Functions


function potential_energy(at:ASEAtoms, tbm::TBModel)
    
    update!(at, tbm)
    
    K, weight = monkhorstpackgrid(atm, tbm)
    E = 0.0
    for n = 1:size(K, 2)
        k = K[:, n]
        epsn_k = get_k_array(tbm, :epsn, k)
        E += weight[n] * r_sum(tbm.smearing(epsn_k, tbm.eF) .* epsn_k)
    end
    
    return E
end





# compute all forces on all the atoms
function forces(atm::AbstractAtoms, tbm::TCTBM)
    # tell tbm to update the spectral decompositions
    update!(atm, tbm)
    
    # allocate output
    Natm = length(atm)
    frc = zeros(3, Natm)

    # precompute neighbourlist
    nlist = NeighbourList(rcut(tbm), atm)

    # BZ integration loop
    K, weight = monkhorstpackgrid(atm, tbm)
    for iK = 1:size(K,2)
        k = K[:, iK]
        epsn_k = get_k_array(tbm, :epsn, k)
        C_k = get_k_array(tbm, :C, k)
        df = @D tbm.smearing(epsn_k)    ##### TODO: HOW DOES SMEARING KNOW eF ????
    
        # loop through all atoms, to compute the force on atm[n]
        for n, neigs, r, R in Sites(nlist)
            # compute the block of indices for the orbitals belonging to n
            In = indexblock(n, tbm)

            # ONSITE TERMS
            # compute ∂H_nn/∂y_n (onsite terms)   >>> DISCUSS WITH HUAJIE
            # IN THE NEW FRAMEWORK THIS SHOULD RETURN A 3-DIMENSIONAL
            # ARRAY WITH ALL THE DERIVATIVES W.R.T. ALL THE SITES!!!
            #  WHY DOES IT LOOK AS IF dH_nn is only norbitals instead of
            #    norbitals x norbitals ????
            
            # dH_nn should be 3 x nneigs x norbitals x norbitals
            dH_nn = @D tbm.onsite(r, R)
            # derivative w.r.t. centre site
            dH_nn_0 = - squeeze(sum(dH_nn, 2), 2)
            for a = 1:tbm.norbitals
                # [ frc[i,n] -= dH_nn[i,a] * dot(df, slice(C, In[a],:).^2) ]
                Ina = In[a]
                t1 = 0.0
                @inbounds @simd for s = 1:length(epsn)
                    t1 += df[s] * C[Ina,s] * C[Ina,s]
                end
                frc[:,n] -= dH_nn_0[:,a] * t1
 	    end
            
            # HOPPING TERMS
            # loop through neighbours of atm[n]
            for i_n = 1:length(neigs)
                m = neigs[i_n]
                Im = indexblock(m, tbm)
                # compute ∂H_nm/∂y_m (hopping terms) and ∂M_nm/∂y_m
                dH_nm = @D tbm.hop(r[i_n], R[:, i_n])
                dM_nm = @D tbm.overlap(r[i,n], R[:, i_n])

                # compute ∂H_nn/∂y_m (onsite terms)   >>>> DICUSS WITH HUAJIE
                #   (WE ALREADY HAVE THIS IN SOME FORM?!)
                #  get_dos!(-r, tbm, dH_nn, ρ)
                
                # the following is a hack to put the on-site assembly into the
                # innermost loop
                
                # add contributions to the force
                #  F_n = ∑_s f'(ϵ_s) < ψ_s | H,n - ϵ_s * M,n | ψ_s >
                for a = 1:tbm.norbitals, b = 1:tbm.norbitals
                    t1 = 0.0; t2 = 0.0; t3 = 0.0
                    ima = Im[a]; inb = In[b]
                    @inbounds @simd for s = 1:length(epsn)
                        t3 = df[s] * C[ima,s] * C[inb,s]
                        t1 += t3
                        t2 += t3 * epsn[s]
                    end
                    frc[:,n] -= 2.0 * (t1 * dH_nm[:,a,b] - t2 * dM_nm[:,a,b])
                end
                for a = 1:tbm.norbitals
                    t1 = 0.0
                    ina = In[a]
                    @inbounds @simd for s = 1:length(epsn)
                        t1 += df[s] * C[ina,s] * C[ina,s]
                    end
                    frc[:,m] -= dH_nn[:,a] * t1
                end
                
            end  # m in neigs-loop
        end  #  sites-loop
    end # k-loop
    
    return frc
end





############################################################
### Site Energy Stuff




##################################################
### MODELS

include("NRLTB.jl")
include("tbtoymodel.jl")


end
