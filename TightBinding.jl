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


include("NRLTB")


"""
`hamiltonian`: compute the Hamiltonian matrix for the tight binding
model.
"""
@protofun hamiltonian(at::AbstractAtoms, tbm::AbstractTBModel)





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
    nkpoints::Vector{T <: Integer}
    
    # internals
    hfd::Float64           # step used for finite-difference approximations
    needupdate::Bool       # tells whether hamiltonian and spectrum are up-to-date
    arrays::Dict{Any, Any}      # storage for various
end

typealias TightBindingModel TBModel


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
    eF
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


monkhorstpackgrid(atm, tbm) = monkhorstpackgrid(cell(atm), tbm.nkpoints)

function monkhorstpackgrid(cell, nkpoints)
    # TODO HUAJIE
end


############################################################
##### update functions

function update_eig!(atm::ASEAtoms, tbm::TBModel, k)
    H, M = hamiltonian(atm, tbm, k)
    epsn, C = sorted_eig(H, M)
    set_k_array!(tbm, epsn, :epsn, k)
    set_k_array!(tbm, C, :C, k)
end


function update_eig!(atm::ASEAtoms, tbm::TBModel)
    K, weight = monkhorstpackgrid(atm, tbm)
    for n = 1:size(K, 2)
        update_eig!(atm, tbm, K[:, n])
    end    
end


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
        update_eF!(atm, tbm)
    end
end



function update_eF!(atm::ASEAtoms, tbm::TBModel)
    if tbm.fixed_eF
        return
    end
    
    # TODO HUAJIE
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
    # non-zeros to make a reasonable first allocation
    #    TODO: ask nlist how much storage we need!
    num_neig_est = length(get_neigs(1, atm)[1])
    nnz_est = tbm.norbitals^2 * num_neig_est * length(atm)
    # allocate space for hamiltonian and overlap matrix
    H = sparse_flexible(nnz_est)
    M = sparse_flexible(nnz_est)
    # OLD: H_nm = zeros(tbm.norbitals, tbm.norbitals)
    # OLD: M_nm = zeros(tbm.norbitals, tbm.norbitals)
    
    # loop through all atoms
    for n, neigs, r, R in Sites(nlist)
        # index-block for atom index n
        In = indexblock(n, tbm)
        # loop through the neighbours of the current atom
        for m = 1:length(M)
            # get the block of indices for atom m
            Im = indexblock(neigs[m], tbm)
            # compute hamiltonian block and add to sparse matrix
            H_nm = tbm.hop(r[m], R[:, m])           #   OLD: get_h!(R[:,m], tbm, H_nm)
            H[In, Im] += H_nm    #  TODO HUAJIE
            # compute overlap block and add to sparse matrix
            M_nm = tbm.overlap(r[m], R[:,m])       #   OLD: get_m!(R[:.m], tbm, M_nm)
            M[In, Im] += M_nm    #  TODO HUAJIE
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



# TODO HUAJIE
# type NRL_Overlap
# end
# evaluate(p::NRL_Overlap, 0.0) = eye(p.norbitals)
# evaluate(p::NRL_Overlap, r, R) = 



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





############################################################
### Site Energy Stuff


end
