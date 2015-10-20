
module ToyTB

#######################################################################
###    The s-orbital Toy model                                      ###
#######################################################################

export ToyTBModel

using Potentials, TightBinding
import Potentials.evaluate, Potentials.evaluate_d
export evaluate, evaluate_d

type ToyTBOverlap <: PairPotential
end

# return 0.0 if two parameters are passed (only for off-diagional terms)
evaluate(p::ToyTBOverlap, r, R) = 0.0
evaluate_d(p::ToyTBOverlap, r, R) = 0.0
# return 1.0 for diagonal terms (when r = 0)
evaluate(p::ToyTBOverlap, r) = (r == 0.0 ? 1.0 : error("ToyTBOverlap(r) : r must be 0.0"))


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
"""
function ToyTBModel(;alpha=2.0, r0=1.0, rcut=2.5, beta=1.0, fixed_eF=true,
                    eF = 0.0, hfd=1e-6)
    
    hop = SWCutoff(MorsePotential(1.0, alpha, r0), rcut, 1.0)
    return TBModel(hop = hop,
                   overlap = ToyTBOverlap(),
                   smearing = FermiDiracSmearing(beta),
                   norbitals = 1,
                   fixed_eF = fixed_eF,
                   eF = eF,
                   nkpoints = (0,0,0),
                   hfd=hfd)
end

end
