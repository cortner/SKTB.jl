



#######################################################################
###    The s-orbital Toy model                                      ###
#######################################################################

type ToyTBOverlap <: PairPotential
end

# return 0.0 if two parameters are passed (only for off-diagional terms)
evaluate(p::ToyTBOverlap, r, R) = 0.0
evaluate_d(p::ToyTBOverlap, r, R) = 0.0
# return 1.0 for diagonal terms (when r = 0)
evaluate(p::ToyTBOverlap, r) = (r == 0.0 ? 1.0 : 0.0)
evaluate_d(p::ToyTBOverlap, r) = 0.0


"""`ToyTBModel`: constructs a simple 1-orbital tight binding model. 
It doesn't model anything but can be used for quick tests. The hopping function
is given by 

  h(r) = MORSE(r; alpha, r0) * η(r; rcut)

### Parameters

* alpha = 2.0, r0 = 1.0, rcut = 2.7  : Morse potential parameters
* beta = 1.0 : electronic temperature
* fixed_eF = true : if true, then the chemical potential is fixed (default at 0.0)
* eF = 0.0 : chemical potential (if fixed)
* hfd = 1e-6 : finite difference step for computing hessians
"""->
function ToyTBModel(;alpha=2.0, r0=1.0, rcut=2.7, beta=1.0, fixed_eF=true,
                    eF = 0.0, hfd=1e-6)
    
    hop = MorsePotential(1.0, A=alpha, r0=r0)
    ### TODO CO: add cut-off !!!!!  , rcut=rcut
    return TBModel(hop = hop,
                   overlap = ToyTBOverlap(),
                   smearing = FermiDiracSmearing(beta),
                   norbitals = 1,
                   fixed_eF = fixed_eF,
                   eF = eF,
                   nkpoints = (0,0,0),
                   hfd=hfd)
end
                  
                  
                  
                  
                  


                  beta, fixedmu, mu, hfd, norbitals, params)
end


# implement the hamiltonian entries
function get_h!(r, tbm::TCTBM{ToyTBModelParameters}, H)
    H[1] = phi(norm(r), tbm.params.hop)
end
function get_dh!(r, tbm::TCTBM{ToyTBModelParameters}, dH)
    dH[:,1,1] = grad_phi(r, tbm.params.hop)
end

function get_m!(r, tbm::TCTBM{ToyTBModelParameters}, M)
    M[:] = 0.0
end
function get_dm!(r, tbm::TCTBM{ToyTBModelParameters}, dM)
    dM[:] = 0.0
end

function get_os!(r, tbm::TCTBM{ToyTBModelParameters}, H)
    H[:] = 0.0
end
function get_dos!(r, tbm::TCTBM{ToyTBModelParameters}, dH, ρ)
    dH[:] = 0.0
end


get_rcut(tbm::TCTBM{ToyTBModelParameters}) = get_rcut(tbm.params.hop)
