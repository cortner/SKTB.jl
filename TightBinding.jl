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
    beta::Float64
    norbitals::Integer
    eF::Float64
    fixed_eF::Bool
    
    # k-point sampling information:
    #    0 = open boundary
    #    1 = Gamma point
    nkpoints::Vector{T <: Integer}
    
    # internals
    hfd::Float64           # step used for finite-difference approximations
    needupdate::Bool       # tells whether hamiltonian and spectrum are up-to-date
    arrays::Dict{Symbol, Any}      # storage for various
end


cutoff(tbm::TBModel) = max(cutoff(tbm.hopping), cutoff(tbm.onsite),
                           cutoff(tbm.pair))


"""`indexblock`:
a little auxiliary function to compute indices for several orbitals
"""
indexblock(n::Union{Integer, Vector}, tbm::TBModel) =
    (n-1) * tbm.norbitals .+ [1:tbm.norbitals;]'



@doc doc"""`hamiltonian`: computes the hamiltonitan and overlap matrix for a tight
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
            H_nm = tbm.hop(r[m])           #   OLD: get_h!(R[:,m], tbm, H_nm)
            H[In, Im] += H_nm    #  TODO HUAJIE
            # compute overlap block and add to sparse matrix
            M_nm = tbm.overlap(r[m])       #   OLD: get_m!(R[:.m], tbm, M_nm)
            M[In, Im] += M_nm    #  TODO HUAJIE
        end
        # now compute the on-site terms
        H_nn = tbm.onsite(r)               #  OLD: get_os!(R, tbm, H_nm)
        H[In, In] += H_nn
        # overlap diagonal block
        M_nn = tbm.overlap(0.0)
        M[In, In] += M_nn
    end

    # convert M, H and return
    return sparse_static(H), sparse_static(M)
end




end
