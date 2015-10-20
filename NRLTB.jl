### TODO : change evaluate_d to grad


module NRLTB

using Potentials

#######################################################################
###      The NRL tight-binding model                                ###
#######################################################################


"""`NRLParams`: collects all the parameters for NRL tight-binding model.
Generally, different element has different parameters.
"""
type NRLParams

# Norbital: 4 if(s, p) : s, px, py, pz
#           9 if(s, p, d) : s, px, py, pz, dxy, dyz, dzx, dx^2-y^2, d3z^2-r^2
# Nbond: 4 if (s, p) : ssσ, spσ, ppσ, ppπ,
#        10 if (s, p ,d) : ssσ, spσ, ppσ, ppπ, sdσ, pdσ, pdπ, ddσ, ddπ, ddδ
    Norbital::Int
    Nbond::Int

# cutoff parameters
    Rc::Float64
    lc::Float64
# onsite
    λ::Float64
    a::Array{Float64}
    b::Array{Float64}
    c::Array{Float64}
    d::Array{Float64}
# hopping H
    e::Array{Float64}
    f::Array{Float64}
    g::Array{Float64}
    h::Array{Float64}
# hopping M
    p::Array{Float64}
    q::Array{Float64}
    r::Array{Float64}
    s::Array{Float64}
end




type NRLos <: SitePotential
    elem :: NRLParams
end
evaluate(p::NRLos, r, R) = get_os(r, elem)
evaluate_d(p::NRLos, r, R) = get_dos(r, R, elem)


type NRLhop <: PairPotential
	elem :: NRLParams
end
evaluate(p::NRLhop, r, R) = mat_local(r, R, p.elem, "H")
evaluate_d(p::NRLhop, r, R) = d_mat_local(r, R, p.elem, "dH")


type NRLoverlap <: PairPotential
	elem :: NRLParams
end
# return 1.0 for diagonal (when r = 0)
evaluate(p::NRLoverlap, r) = (r == 0.0 ? 
	eye(p.elem.Norbital) : error("NRLoverlap(r) may only be called with r = 0.0") )
evaluate_d(p::NRLoverlap, r, R) = 
		zeros(3, p.elem.Norbital, p.elem.Norbital)
# off-diagonal terms
evaluate(p::NRLoverlap, r, R) = mat_local(r, R, p.elem, "M")
evaluate_d(p::NRLoverlap, r, R) = d_mat_local(r, R, p.elem, "dM")




"""`NRLTBModel`: constructs the NRL tight binding model. 

### Parameters

* elem : NRLParams (default at Carbon atom with s&p orbitals)
* beta = 1.0 : electronic temperature
* fixed_eF = true : if true, then the chemical potential is fixed (default at 0.0)
* eF = 0.0 : chemical potential (if fixed)
* nkpoints : number of k-points at each direction (only (0,0,Int) has been implemented)
* hfd = 1e-6 : finite difference step for computing hessians
"""
function NRLTBModel(; elem = C_sp, beta=1.0, fixed_eF=true, eF = 0.0, 
		    nkpoints = (0, 0, 0), hfd=1e-6)

    onsite = NRLos(elem)
    hop  = NRLhop(elem)
    overlpa = NRLoverlap(elem)

    return TBModel(onsite = onsite,
		   hop = hop,
                   overlap = overlap,
                   smearing = FermiDiracSmearing(beta),
                   norbitals = elem.Norbital,
                   fixed_eF = fixed_eF,
                   eF = eF,
                   nkpoints = nkpoints,
                   hfd=hfd)
end




########################################################
###  NRL Cut-off
### TODO: at some point apply the nice `Potentials` abstractions!
# type NRLCutoff <: CutoffPotential
#     pp::PairPotential
#     Lc::Float64
#     Rc::Float64
#     Mc::Float64   # NRL wants 5.0, but we take 10.0 to be on the "safe side"
# end
# evaluate(p::NRLCutoff, r) =
#     ( (1.0 ./ (1.0 + exp( (r-p.Rc) / p.Lc + p.Mc )) - 1.0 ./ (1.0 + exp(p.Mc)))
#       .* (r .<= p.Rc) )

# nrlcutoff_d(r) = 
#     ( (1.0 ./ (1.0 + exp( (r-p.Rc) / p.Lc + p.Mc )) - 1.0 ./ (1.0 + exp(p.Mc)))
#       .* (r .<= p.Rc) )


####################################################################################
############ Some functions to construct Hamiltonian and Overlap Matrices ##########
####################################################################################

# (Modified) cutoff function in the NRL ansatz for Hamiltonian matrix elements
#    r  : variable
#    Rc : cutoff radius
#    lc : cutoff weight
#    Mc : change NRL's 5.0 to 10.0

function cutoff_NRL(r, Rc, lc; Mc=10.0)
    fcut = (1.0 ./ (1.0 + exp( (r-Rc) / Lc + Mc )) - 1.0 ./ (1.0 + exp(Mc))) .* (r .<= Rc) 
    return fcut
end

# first order derivative
function d_cutoff_NRL(r, Rc, lc; Mc=10.0)
    temp = exp( (r-Rc) ./ lc + Mc )
    d_fcut = - 1.0 ./ ( 1.0 + temp ).^2 .* temp ./ lc .* (r .<= Rc)
    return d_fcut
end


# Pseudo electron density on site l : ρ_l
# Input
# r, R : distances and displacements of the neighboring atoms 
# elem : NRLParams 
# OUTPUT
# ρ    : return the pseudo density on site n = 1, ... , length(atm)

function pseudoDensity(r::Vector{Float64}, elem::NRLParams)
    λ = elem.λ
    Rc = elem.Rc
    lc = elem.lc
    # dX = Float64[ norm(r[:,k])  for k = 1:size(r,2) ]
    dX = Float64[ r[k]  for k = 1:length(r) ]
    eX = exp(-(λ^2) * dX)
    fX = cutoff_NRL(dX, Rc, lc)
    # note that the NRL pseudo density has ignored the self-distance
    ρ = sum( eX .* fX ) 

    return ρ
end

# first order derivative, return:  dρ_l / d R_lk
function dR_pseudoDensity(Rlk, elem::NRLParams)
    λ = elem.λ
    Rc = elem.Rc
    lc = elem.lc
    cR = cutoff_NRL(Rlk, Rc, lc)
    dcR = d_cutoff_NRL(Rlk, Rc, lc)
    ρ_dR = exp(-(λ^2) * Rlk) * ( -λ^2 * cR + dcR )
    return ρ_dR
end


# auxiliary functions for computing the onsite terms
function os_NRL(elem::NRLParams, ρ::Float64)
    a = elem.a
    b = elem.b
    c = elem.c
    d = elem.d
    hl = Float64[ a[i] + b[i] * ρ^(2/3) + c[i] * ρ^(4/3) + d[i] * ρ^2   
				for i=1:elem.Norbital ]
    return hl
end

function os_NRL(elem::NRLParams, ρ::Vector{Float64})
    a = elem.a
    b = elem.b
    c = elem.c
    d = elem.d
    hl = Float64[ a[i] + b[i] * ρ[j]^(2/3) + c[i] * ρ[j]^(4/3) + d[i] * ρ[j]^2   
				for i=1:elem.Norbital, j=1:length(ρ) ]
    return hl
end

# first order derivative, return : d h_os / d ρ
function dρ_os_NRL(elem::NRLParams, ρ::Float64)
    a = elem.a
    b = elem.b
    c = elem.c
    d = elem.d
    hl = Float64[ 2/3 * b[i] * ρ^(-1/3) + 4/3 * c[i] * ρ^(1/3) + 2 * d[i] * ρ   
				for i=1:elem.Norbital ]
    return hl
end

function dρ_os_NRL(elem::NRLParams, ρ::Vector{Float64})
    a = elem.a
    b = elem.b
    c = elem.c
    d = elem.d
    hl = Float64[ 2/3 * b[i] * ρ[j]^(-1/3) + 4/3 * c[i] * ρ[j]^(1/3) + 2 * d[i] * ρ[j]   
				for i=1:elem.Norbital, j=1:length(ρ) ]
    return hl
end

# get the onsite terms
function get_os(r::Vector{Float64}, elem::NRLParams)
	n = elem.Norbital
    H = zeros(n, n)
    ρ = pseudoDensity(r, elem)
    h = os_NRL(elem, ρ)
    for i = 1:n
        H[i,i] = h[i]
    end
	return H
end

# first order derivative
function get_dos(r::Vector{Float64}, R::Array{Float64}, elem::NRLParams)
	dim = 3
	nneig = length(r) 
	norbitals = elem.Norbital 	
        dH = zeros(dim, norbitals, norbitals, nneig)
	# compute ∂H_nn/∂y_m ;
	for m = 1:nneig
            dρ = dR_pseudoDensity(r[m], elem)
            dh = dρ_os_NRL(elem, ρ)
            for d = 1:dim, i = 1:norbitals
                dH[d, m+1, i, i] = dρ * dh[i] * R[d,m]/r[m]
            end
	end
	return dH
end




## hopping terms in H
# INPUT
# R         : |atom[l] - atom[k]|
# bond_type : αβγ
# elem	    : NRLParams

function h_hop(R, bond_type, elem::NRLParams)
    Rc = elem.Rc
    lc = elem.lc
    e = elem.e[bond_type]
    f = elem.f[bond_type]
    g = elem.g[bond_type]
    h = elem.h[bond_type]
    hαβγ = (e + f*R + g*R^2) * exp(-h^2*R) * cutoff_NRL(R, Rc, lc)
    return hαβγ
end

# first order derivative
function dR_h_hop(R, bond_type, elem::NRLParams)
    Rc = elem.Rc
    lc = elem.lc
    e = elem.e[bond_type]
    f = elem.f[bond_type]
    g = elem.g[bond_type]
    h = elem.h[bond_type]
    cR = cutoff_NRL(R, Rc, lc)
    dcR = d_cutoff_NRL(R, Rc, lc)
    hαβγ = exp(-h^2*R) * ( (f + 2*g*R) * cR - 
		h^2 * (e + f*R + g*R^2) * cR + (e + f*R + g*R^2) * dcR )
    return hαβγ
end



## hopping terms in M(OVERLAP)

function m_hop(R, bond_type, elem::NRLParams)
    Rc = elem.Rc
    lc = elem.lc
    p = elem.p[bond_type]
    q = elem.q[bond_type]
    r = elem.r[bond_type]
    s = elem.s[bond_type]
    mαβγ = (p + q*R + r*R^2) * exp(-s^2*R) * cutoff_NRL(R, Rc, lc)
    return mαβγ
end

# first order derivative
function dR_m_hop(R, bond_type, elem::NRLParams)
    Rc = elem.Rc
    lc = elem.lc
    p = elem.p[bond_type]
    q = elem.q[bond_type]
    r = elem.r[bond_type]
    s = elem.s[bond_type]
    cR = cutoff_NRL(R, Rc, lc)
    dcR = d_cutoff_NRL(R, Rc, lc)
    mαβγ = exp(-s^2*R) * ( (q + 2*r*R) * cR - 
			s^2 * (p + q*R + r*R^2) * cR + 
			(p + q*R + r*R^2) * dcR )
    return mαβγ
end



## generates local hamiltonian and overlap for hopping terms or overlap.
# The size of returnned local matrices are  Norbit x Norbit,
# for example, 4x4 for s&p orbitals and 9x9 for s&p&d orbitals.
# INPUT
# r: distance
# R: displacement
# elem::NRLParams
# task: may be { "H", "M" }
# OUTPUT
# h : R^{norb x norb}

function mat_local(r::Float64, R::Vector{Float64}, elem::NRLParams, task)
    # r = norm(R)
    u = R/r
    dim == 3
    l,m,n = u[:]
    Norb = elem.Norbital
    Nb = elem.Nbond
    h = zeros(Norb, Norb)

    # use different functions for different tasks
    if task == "H"
        hh = Float64[ h_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
    elseif task == "M"
        hh = Float64[ m_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
    else
        throw(ArgumentError("this task has not been implemented yet"))
    end
	
    if Norb == 4 && Nb == 4
    # 4 orbitals are s,px,py,pz; 4 bond types are : ssσ,spσ,ppσ,ppπ
        h[1,1] = hh[1]
        h[2,2] = l*l * hh[3] + (1.-l*l) * hh[4]
        h[3,3] = m*m * hh[3] + (1.-m*m) * hh[4]
        h[4,4] = n*n * hh[3] + (1.-n*n) * hh[4]
        h[1,2] = l * hh[2]
        h[1,3] = m * hh[2]
        h[1,4] = n * hh[2]
        h[2,1] = - h[1,2]
        h[3,1] = - h[1,3]
        h[4,1] = - h[1,4]
        h[2,3] = l * m * (hh[3] - hh[4])
        h[2,4] = l * n * (hh[3] - hh[4])
        h[3,4] = m * n * (hh[3] - hh[4])
        h[3,2] =  h[2,3]
        h[4,2] =  h[2,4]
        h[4,3] =  h[3,4]

    elseif Norb == 9 && Nb == 10
    # 9 orbitals : s, px, py, pz, dxy, dyz, dzx, dx2-y2, d3z2-r2
    # 10 bond types are : 1ssσ, 2spσ, 3ppσ, 4ppπ, 5sdσ, 6pdσ, 7pdπ, 8ddσ, 9ddπ, 10ddδ
        
        # ss
        h[1,1] = hh[1]

        # sp
        h[1,2] = l * hh[2]
        h[1,3] = m * hh[2]
        h[1,4] = n * hh[2]
        h[2,1] = - h[1,2]
        h[3,1] = - h[1,3]
        h[4,1] = - h[1,4]

        # pp
        h[2,2] = l^2 * hh[3] + (1-l^2) * hh[4]
        h[3,3] = m^2 * hh[3] + (1-m^2) * hh[4]
        h[4,4] = n^2 * hh[3] + (1-n^2) * hh[4]
        h[2,3] = l * m * (hh[3] - hh[4])
        h[2,4] = l * n * (hh[3] - hh[4])
        h[3,4] = m * n * (hh[3] - hh[4])
        h[3,2] =  h[2,3]
        h[4,2] =  h[2,4]
        h[4,3] =  h[3,4]

        # sd
        h[1,5] = √3 * l * m * hh[5]
        h[1,6] = √3 * m * n * hh[5]
        h[1,7] = √3 * l * n * hh[5]
        h[1,8] = √3/2 * (l^2 - m^2) * hh[5]
        h[1,9] = ( n^2 - (l^2 + m^2)/2 ) * hh[5]
        h[5,1] = h[1,5]
        h[6,1] = h[1,6]
        h[7,1] = h[1,7]
        h[8,1] = h[1,8]
        h[9,1] = h[1,9]
        
        # pd
        h[2,5] = √3 * l * l * m * hh[6] + m * (1.0 - 2.0 * l^2) * hh[7]
        h[2,6] = √3 * l * m * n * hh[6] - 2.0 * l * m * n * hh[7]
        h[2,7] = √3 * l * l * n * hh[6] + n * (1.0 - 2.0 * l^2) * hh[7]
        h[2,8] = √3/2 * l * (l^2 - m^2) * hh[6] + l * (1.0 - l^2 + m^2) * hh[7]
        h[2,9] = l * (n^2 - (l^2 + m^2)/2) * hh[6] - √3 * l * n^2 * hh[7]
        h[5,2] = - h[2,5]
        h[6,2] = - h[2,6]
        h[7,2] = - h[2,7]
        h[8,2] = - h[2,8]
        h[9,2] = - h[2,9]
        h[3,5] = √3 * l * m * m * hh[6] + l * (1.0 - 2.0 * m^2) * hh[7]
        h[3,6] = √3 * m * m * n * hh[6] + n * (1.0 - 2.0 * m^2) * hh[7]
        h[3,7] = √3 * l * m * n * hh[6] - 2.0 * l * m * n * hh[7]
        h[3,8] = √3/2 * m * (l^2 - m^2) * hh[6] - m * (1.0 + l^2 - m^2) * hh[7]
        h[3,9] = m * (n^2 - (l^2 + m^2)/2) * hh[6] - √3 * m * n^2 * hh[7]
        h[5,3] = - h[3,5]
        h[6,3] = - h[3,6]
        h[7,3] = - h[3,7]
        h[8,3] = - h[3,8]
        h[9,3] = - h[3,9]
        h[4,5] = √3 * l * m * n * hh[6] - 2.0 * l * m * n * hh[7]
        h[4,6] = √3 * m * n * n * hh[6] + m * (1.0 - 2.0 * n^2) * hh[7]
        h[4,7] = √3 * l * n * n * hh[6] + l * (1.0 - 2.0 * n^2) * hh[7]
        h[4,8] = √3/2 * n * (l^2 - m^2) * hh[6] - n * (l^2 - m^2) * hh[7]
        h[4,9] = n * (n^2 - (l^2 + m^2)/2) * hh[6] + √3 * n * (l^2 + m^2) * hh[7]
        h[5,4] = - h[4,5]
        h[6,4] = - h[4,6]
        h[7,4] = - h[4,7]
        h[8,4] = - h[4,8]
        h[9,4] = - h[4,9]

        # dd
        h[5,5] = 3.0 * l^2 * m^2 * hh[8] + (l^2 + m^2 - 4.0 * l^2 * m^2) * hh[9] + (n^2 + l^2 * m^2) * hh[10]
        h[6,6] = 3.0 * m^2 * n^2 * hh[8] + (m^2 + n^2 - 4.0 * m^2 * n^2) * hh[9] + (l^2 + m^2 * n^2) * hh[10]
        h[7,7] = 3.0 * l^2 * n^2 * hh[8] + (l^2 + n^2 - 4.0 * l^2 * n^2) * hh[9] + (m^2 + l^2 * n^2) * hh[10]
        h[8,8] = 3.0/4 * (l^2 - m^2)^2 * hh[8] + (l^2 + m^2 - (l^2 - m^2)^2) * hh[9] + 
		 (n^2 + (l^2 - m^2)^2 /4 ) * hh[10]
        h[9,9] = (n^2 - (l^2 + m^2) /2)^2 * hh[8] + 3.0 * n^2 * (l^2 + m^2) * hh[9] + 
		 3.0/4 * (l^2 + m^2)^2 * hh[10]
        h[5,6] = 3.0 * l * m^2 * n * hh[8] + l * n * (1.0 - 4.0 * m^2) * hh[9] + 
		 l * n * (m^2 - 1.0) * hh[10]
        h[5,7] = 3.0 * l^2 * m * n * hh[8] + m * n * (1.0 - 4.0 * l^2) * hh[9] + 
		 m * n * (l^2 - 1.0) * hh[10]
        h[5,8] = 3.0/2 * l * m * (l^2 - m^2) * hh[8] + 2.0 * l * m * (m^2 - l^2) * hh[9] + 
		 1.0/2 * l * m * (l^2 - m^2) * hh[10]
        h[5,9] = √3 * l * m * (n^2 - (l^2 + m^2)/2) * hh[8] - 2.0*√3 * l * m * n^2 * hh[9] + 
		 √3/2 * l * m * (1.0 + n^2) * hh[10]
        h[6,7] = 3.0 * l * m * n^2 * hh[8] + l * m * (1.0 - 4.0 * n^2) * hh[9] + 
		 l * m * (n^2 - 1.0) * hh[10]
        h[6,8] = 3.0/2 * m * n * (l^2 - m^2) * hh[8] - m * n * (1.0 + 2.0 * (l^2 - m^2)) * hh[9] + 
		 m * n * (1.0 + (l^2 - m^2) /2) * hh[10]
        h[6,9] = √3 * m * n * (n^2 - (l^2 + m^2) /2) * hh[8] + √3 * m * n * (l^2 + m^2 - n^2) * hh[9] - 
		 √3/2 * m * n * (l^2 + m^2) * hh[10]
        h[7,8] = 3.0/2 * l * n * (l^2 - m^2) * hh[8] + l * n * (1.0 - 2.0 * (l^2 - m^2)) * hh[9] - 
		 l * n * (1.0 - (l^2 - m^2) /2) * hh[10]        
        h[7,9] = √3 * l * n * (n^2 - (l^2 + m^2) /2) * hh[8] + √3 * l * n * (l^2 + m^2 - n^2) * hh[9] - 
		 √3/2 * l * n * (l^2 + m^2) * hh[10]
        h[8,9] = √3/2 * (l^2 - m^2) * (n^2 - (l^2 + m^2) /2) * hh[8] + √3 * n^2 * (m^2 - l^2) * hh[9] + 
		 √3/4 * (1.0 + n^2) * (l^2 - m^2) * hh[10]         
        h[6,5] = h[5,6]
        h[7,5] = h[5,7]
        h[8,5] = h[5,8]
        h[9,5] = h[5,9]
        h[7,6] = h[6,7]
        h[8,6] = h[6,8]
        h[9,6] = h[6,9]
        h[8,7] = h[7,8]
        h[9,7] = h[7,9]
        h[9,8] = h[8,9]

    else
        throw(ArgumentError("the numbers of atomic orbitals and bond types do not match!"))
    end
    return h
end



# generates local matrix for dH and dM

function d_mat_local(r::Float64, RR::Vector{Float64}, elem::NRLParams, task)
    #r = norm(RR)
    u = RR/r
    dim == 3
    l,m,n = u[:]
    Norb = elem.Norbital
    Nb = elem.Nbond
    dh = zeros(dim, Norb, Norb)
    # dR/dx = x/R = l, dR/dy = y/R = m, dR/dz = z/R = n
    # u = [l,m,n], l = x/R, m = y/R, n = z/R
    # dl/dx = 1/R - x^2/R^3 = 1/R - l^2/R, dl/dy = -xy/R^3 = -lm/R, dl/dz = -xz/R^3 = -ln/R
    R = r
    dR = [l, m, n]  
    dl = [1/R - l*l/R , - l*m/R , - l*n/R]
    dm = [- l*m/R , 1/R - m*m/R , - m*n/R]
    dn = [- l*n/R , - m*n/R , 1/R - n*n/R]

    # use different functions for different tasks
    if task == "dH"
        hh = Float64[ h_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
        dhh = Float64[ dR_h_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
    elseif task == "dM"
        hh = Float64[ m_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
        dhh = Float64[ dR_m_hop(r, bond_type, elem)  for bond_type = 1:Nb ]
    else
        throw(ArgumentError("this task has not been implemented yet"))
    end

    if Norb == 4 && Nb == 4
        # 4 orbitals are s,px,py,pz; 4 bond types are : ssσ,spσ,ppσ,ppπ, now calculate all the derivatives
        # dh/dx = dh/dR * dR/dx + dh/dl * dl/dx + dh/dm * dm/dx + dh/dn * dn/dx
        # similarly dh/dy and dh/dz
       for d = 1 : dim
            dh[d,1,1] = dhh[1] * dR[d]
            dh[d,2,2] = l*l * dhh[3] * dR[d] + (1.-l*l) * dhh[4] * dR[d] + 2*l * hh[3] * dl[d] - 2*l * hh[4] * dl[d]
            dh[d,3,3] = m*m * dhh[3] * dR[d] + (1.-m*m) * dhh[4] * dR[d] + 2*m * hh[3] * dm[d] - 2*m * hh[4] * dm[d]
            dh[d,4,4] = n*n * dhh[3] * dR[d] + (1.-n*n) * dhh[4] * dR[d] + 2*n * hh[3] * dn[d] - 2*n * hh[4] * dn[d]
            dh[d,1,2] = l * dhh[2] * dR[d] + hh[2] * dl[d]
            dh[d,1,3] = m * dhh[2] * dR[d] + hh[2] * dm[d]
            dh[d,1,4] = n * dhh[2] * dR[d] + hh[2] * dn[d]
            dh[d,2,1] = - dh[d,1,2]
            dh[d,3,1] = - dh[d,1,3]
            dh[d,4,1] = - dh[d,1,4]
            dh[d,2,3] = l * m * (dhh[3] - dhh[4]) * dR[d] + (dl[d] * m + l * dm[d]) * (hh[3] - hh[4])
            dh[d,2,4] = l * n * (dhh[3] - dhh[4]) * dR[d] + (dl[d] * n + l * dn[d]) * (hh[3] - hh[4])
            dh[d,3,4] = m * n * (dhh[3] - dhh[4]) * dR[d] + (dm[d] * n + m * dn[d]) * (hh[3] - hh[4])
            dh[d,3,2] =  dh[d,2,3]
            dh[d,4,2] =  dh[d,2,4]
            dh[d,4,3] =  dh[d,3,4]
        end        

    elseif Norb == 9 && Nb == 10
        # 9 orbitals : s, px, py, pz, dxy, dyz, dzx, dx2-y2, d3z2-r2
        for d = 1 : dim
            # ss
            dh[d,1,1] = dhh[1] * dR[d]
            # sp
            dh[d,1,2] = l * dhh[2] * dR[d] + hh[2] * dl[d]
            dh[d,1,3] = m * dhh[2] * dR[d] + hh[2] * dm[d]
            dh[d,1,4] = n * dhh[2] * dR[d] + hh[2] * dn[d]
            dh[d,2,1] = - dh[d,1,2]
            dh[d,3,1] = - dh[d,1,3]
            dh[d,4,1] = - dh[d,1,4]
            # pp
            dh[d,2,2] = l*l * dhh[3] * dR[d] + (1.-l*l) * dhh[4] * dR[d] + 2*l * hh[3] * dl[d] - 2*l * hh[4] * dl[d]
            dh[d,3,3] = m*m * dhh[3] * dR[d] + (1.-m*m) * dhh[4] * dR[d] + 2*m * hh[3] * dm[d] - 2*m * hh[4] * dm[d]
            dh[d,4,4] = n*n * dhh[3] * dR[d] + (1.-n*n) * dhh[4] * dR[d] + 2*n * hh[3] * dn[d] - 2*n * hh[4] * dn[d]            
            dh[d,2,3] = l * m * (dhh[3] - dhh[4]) * dR[d] + (dl[d] * m + l * dm[d]) * (hh[3] - hh[4])
            dh[d,2,4] = l * n * (dhh[3] - dhh[4]) * dR[d] + (dl[d] * n + l * dn[d]) * (hh[3] - hh[4])
            dh[d,3,4] = m * n * (dhh[3] - dhh[4]) * dR[d] + (dm[d] * n + m * dn[d]) * (hh[3] - hh[4])
            dh[d,3,2] =  dh[d,2,3]
            dh[d,4,2] =  dh[d,2,4]
            dh[d,4,3] =  dh[d,3,4]
            # sd
            dh[d,1,5] = √3 * l * m * dhh[5] * dR[d] + √3 * ( dl[d] * m + l * dm[d] ) * hh[5]
            dh[d,1,6] = √3 * m * n * dhh[5] * dR[d] + √3 * ( dm[d] * n + m * dn[d] ) * hh[5]
            dh[d,1,7] = √3 * l * n * dhh[5] * dR[d] + √3 * ( dl[d] * n + l * dn[d] ) * hh[5]
            dh[d,1,8] = √3/2 * (l^2 - m^2) * dhh[5] * dR[d] + √3/2 * ( 2*l * dl[d] - 2*m * dm[d]) * hh[5]
            dh[d,1,9] = ( n^2 - (l^2 + m^2)/2 ) * dhh[5] * dR[d] + ( 2*n * dn[d] - (l * dl[d] + m * dm[d]) ) * hh[5]
            dh[d,5,1] = dh[d,1,5]
            dh[d,6,1] = dh[d,1,6]
            dh[d,7,1] = dh[d,1,7]
            dh[d,8,1] = dh[d,1,8]
            dh[d,9,1] = dh[d,1,9]
            # pd
            dh[d,2,5] = √3 * l * l * m * dhh[6] * dR[d] + m * (1.0 - 2.0 * l^2) * dhh[7] * dR[d] + 
                        √3 * (2*l * m * dl[d] + l^2 * dm[d]) * hh[6] + 
                        ( dm[d] * (1.0 - 2.0 * l^2) - 4*m*l * dl[d] ) * hh[7]
            dh[d,2,6] = √3 * l * m * n * dhh[6] * dR[d] - 2.0 * l * m * n * dhh[7] * dR[d] + 
                        √3 * (dl[d] * m * n + l * dm[d] * n + l * m * dn[d]) * hh[6] - 
                        2.0 * (dl[d] * m * n + l * dm[d] * n + l * m * dn[d]) * hh[7]
            dh[d,2,7] = √3 * l * l * n * dhh[6] * dR[d] + n * (1.0 - 2.0 * l^2) * dhh[7] * dR[d] + 
                        √3 * (2*l * dl[d] * n + l*l * dn[d]) * hh[6] + 
                        ( dn[d] * (1.0 - 2.0 * l^2) - 4*l*n * dl[d] ) * hh[7]
            dh[d,2,8] = √3/2 * l * (l^2 - m^2) * dhh[6] * dR[d] + l * (1.0 - l^2 + m^2) * dhh[7] * dR[d] + 
                        √3/2 * (dl[d] * (l^2 - m^2) + l * (2*l * dl[d] - 2*m * dm[d])) * hh[6] + 
                        (dl[d] * (1.0 - l^2 + m^2) + l * (-2*l * dl[d] + 2*m * dm[d]) ) * hh[7]
            dh[d,2,9] = l * (n^2 - (l^2 + m^2)/2) * dhh[6] * dR[d] - √3 * l * n^2 * dhh[7] * dR[d] + 
                        ( dl[d] * (n^2 - (l^2 + m^2)/2) + l * (2*n * dn[d] - l * dl[d] - m * dm[d]) ) * hh[6] - 
                        √3 * (dl[d] * n^2 + 2*l*n * dn[d])* hh[7]
            dh[d,5,2] = - dh[d,2,5]
            dh[d,6,2] = - dh[d,2,6]
            dh[d,7,2] = - dh[d,2,7]
            dh[d,8,2] = - dh[d,2,8]
            dh[d,9,2] = - dh[d,2,9]
            dh[d,3,5] = √3 * l * m * m * dhh[6] * dR[d] + l * (1.0 - 2.0 * m^2) * dhh[7] * dR[d] + 
                        √3 * (dl[d] * m^2 + 2*l*m * dm[d]) * hh[6] + 
                        ( dl[d] * (1.0 - 2.0 * m^2) - 4*l*m * dm[d] ) * hh[7]
            dh[d,3,6] = √3 * m * m * n * dhh[6] * dR[d] + n * (1.0 - 2.0 * m^2) * dhh[7] * dR[d] + 
                        √3 * (2*m*n * dm[d] + m^2 * dn[d]) * hh[6] + 
                        ( dn[d] * (1.0 - 2.0 * m^2) - 4*m*n * dm[d] ) * hh[7]
            dh[d,3,7] = √3 * l * m * n * dhh[6] * dR[d] - 2.0 * l * m * n * dhh[7] * dR[d] +  
                        ( dl[d] * m * n + l * dm[d] * n + l * m * dn[d] ) * ( √3 * hh[6] - 2.0 * hh[7] )
            dh[d,3,8] = √3/2 * m * (l^2 - m^2) * dhh[6] * dR[d] - m * (1.0 + l^2 - m^2) * dhh[7] * dR[d] +  
                        √3/2 * ( dm[d] * (l^2 - m^2) + m * (2*l * dl[d] - 2*m * dm[d]) ) * hh[6] - 
                        ( dm[d] * (1.0 + l^2 - m^2) + m * (2*l * dl[d] - 2*m * dm[d]) ) * hh[7]
            dh[d,3,9] = m * (n^2 - (l^2 + m^2)/2) * dhh[6] * dR[d] - √3 * m * n^2 * dhh[7] * dR[d] + 
                        ( dm[d] * (n^2 - (l^2 + m^2)/2) + m * (2*n * dn[d] - l * dl[d] - m * dm[d]) ) * hh[6] - 
                        √3 * ( dm[d] * n^2 + 2*m*n * dn[d] ) * hh[7]
            dh[d,5,3] = - dh[d,3,5]
            dh[d,6,3] = - dh[d,3,6]
            dh[d,7,3] = - dh[d,3,7]
            dh[d,8,3] = - dh[d,3,8]
            dh[d,9,3] = - dh[d,3,9]
            dh[d,4,5] = √3 * l * m * n * dhh[6] * dR[d] - 2.0 * l * m * n * dhh[7] * dR[d] +  
                        ( dl[d] * m * n + l * dm[d] * n + l * m * dn[d] ) * ( √3 * hh[6] - 2.0 * hh[7] )
            dh[d,4,6] = √3 * m * n * n * dhh[6] * dR[d] + m * (1.0 - 2.0 * n^2) * dhh[7] * dR[d] +  
                        √3 * ( dm[d] * n^2 + 2*m*n * dn[d] ) * hh[6] + 
                        ( dm[d] * (1.0 - 2.0 * n^2) - 4*m*n * dn[d] ) * hh[7]
            dh[d,4,7] = √3 * l * n * n * dhh[6] * dR[d] + l * (1.0 - 2.0 * n^2) * dhh[7] * dR[d] +  
                        √3 * ( dl[d] * n^2 + 2*l*n * dn[d] ) * hh[6] + 
                        ( dl[d] * (1.0 - 2.0 * n^2) - 4*l*n * dn[d] ) * hh[7]
            dh[d,4,8] = √3/2 * n * (l^2 - m^2) * dhh[6] * dR[d] - n * (l^2 - m^2) * dhh[7] * dR[d] + 
                        √3/2 * ( dn[d] * (l^2 - m^2) + n * (2*l * dl[d] - 2*m * dm[d]) ) * hh[6] - 
                        ( dn[d] * (l^2 - m^2) + n * (2*l * dl[d] - 2*m * dm[d]) ) * hh[7]
            dh[d,4,9] = n * (n^2 - (l^2 + m^2)/2) * dhh[6] * dR[d] + √3 * n * (l^2 + m^2) * dhh[7] * dR[d] + 
                        ( dn[d] * (n^2 - (l^2 + m^2)/2) +  n * (2*n * dn[d] - l * dl[d] - m * dm[d]) ) * hh[6] + 
                        √3 * ( dn[d] * (l^2 + m^2) + n * (2*l * dl[d] + 2*m * dm[d]) ) * hh[7]
            dh[d,5,4] = - dh[d,4,5]
            dh[d,6,4] = - dh[d,4,6]
            dh[d,7,4] = - dh[d,4,7]
            dh[d,8,4] = - dh[d,4,8]
            dh[d,9,4] = - dh[d,4,9]                
            # dd
            dh[d,5,5] = 3.0 * l^2 * m^2 * dhh[8] * dR[d] + (l^2 + m^2 - 4.0 * l^2 * m^2) * dhh[9] * dR[d] + 
						(n^2 + l^2 * m^2) * dhh[10] * dR[d] +  
						3.0 * ( 2*l * dl[d] * m^2 + l^2 * 2*m * dm[d] ) * hh[8] + 
                        (2*l * dl[d] + 2*m * dm[d] - 8*l * dl[d] * m^2 - 8*m * l^2 * dm[d]) * hh[9] + 
                        (2*n * dn[d] + 2*l * dl[d] * m^2 + 2*m * l^2 * dm[d]) * hh[10]
            dh[d,6,6] = 3.0 * m^2 * n^2 * dhh[8] * dR[d] + (m^2 + n^2 - 4.0 * m^2 * n^2) * dhh[9] * dR[d] + 
						(l^2 + m^2 * n^2) * dhh[10] * dR[d] + 
                        3.0 * ( 2*m * dm[d] * n^2 + m^2 * 2*n * dn[d] ) * hh[8] + 
                        (2*m * dm[d] + 2*n * dn[d] - 8*m * dm[d] * n^2 - 8*n * m^2 * dn[d]) * hh[9] + 
                        (2*l * dl[d] + 2*m * dm[d] * n^2 + 2*n * m^2 * dn[d]) * hh[10]
            dh[d,7,7] = 3.0 * l^2 * n^2 * dhh[8] * dR[d] + (l^2 + n^2 - 4.0 * l^2 * n^2) * dhh[9] * dR[d] + 
						(m^2 + l^2 * n^2) * dhh[10] * dR[d] + 
                        3.0 * ( 2*l * dl[d] * n^2 + l^2 * 2*n * dn[d] ) * hh[8] + 
                        (2*l * dl[d] + 2*n * dn[d] - 8*l * dl[d] * n^2 - 8*n * l^2 * dn[d]) * hh[9] + 
                        (2*m * dm[d] + 2*l * dl[d] * n^2 + 2*n * l^2 * dn[d]) * hh[10]
            dh[d,8,8] = 3.0/4 * (l^2 - m^2)^2 * dhh[8] * dR[d] + (l^2 + m^2 - (l^2 - m^2)^2) * dhh[9] * dR[d] + 
						(n^2 + (l^2 - m^2)^2 /4) * dhh[10] * dR[d] + 
                        3.0 * (l^2 - m^2) * (l * dl[d] - m * dm[d]) * hh[8] + 
                        ( 2*l * dl[d] + 2*m * dm[d] - 4.0 * (l^2 - m^2) * (l * dl[d] - m * dm[d]) ) * hh[9] + 
                        ( 2*n * dn[d] + (l^2 - m^2) * (l * dl[d] - m * dm[d]) ) * hh[10]
            dh[d,9,9] = (n^2 - (l^2 + m^2) /2)^2 * dhh[8] * dR[d] + 3.0 * n^2 * (l^2 + m^2) * dhh[9] * dR[d] + 
						3.0/4 * (l^2 + m^2)^2 * dhh[10] * dR[d] + 
                        2.0 * (n^2 - (l^2 + m^2) /2) * (2*n * dn[d] - (l * dl[d] + m * dm[d])) * hh[8] + 
                        3.0 * ( 2*n * dn[d] * (l^2 + m^2) + n^2 * (2*l * dl[d] + 2*m * dm[d]) ) * hh[9] + 
                        3.0 * (l^2 + m^2) * (l * dl[d] + m * dm[d]) * hh[10]     
            dh[d,5,6] = 3.0 * l * m^2 * n * dhh[8] * dR[d] + l * n * (1.0 - 4.0 * m^2) * dhh[9] * dR[d] + 
						l * n * (m^2 - 1.0) * dhh[10] * dR[d] + 
                        3.0 * (dl[d] * m^2 * n + l * 2*m * dm[d] * n + l * m^2 * dn[d]) * hh[8] + 
                        ( dl[d] * n * (1.0 - 4.0 * m^2) + l * dn[d] * (1.0 - 4.0 * m^2) - l * n * 8*m * dm[d] ) * hh[9] + 
                        ( dl[d] * n * (m^2 - 1.0) + l * dn[d] * (m^2 - 1.0) + l * n * 2*m * dm[d] ) * hh[10]
            dh[d,5,7] = 3.0 * l^2 * m * n * dhh[8] * dR[d] + m * n * (1.0 - 4.0 * l^2) * dhh[9] * dR[d] + 
						m * n * (l^2 - 1.0) * dhh[10] * dR[d] + 
                        3.0 * (2*l * dl[d] * m * n + l^2 * dm[d] * n + l^2 * m * dn[d]) * hh[8] + 
                        ( dm[d] * n * (1.0 - 4.0 * l^2) + m * dn[d] * (1.0 - 4.0 * l^2) - m * n * 8*l * dl[d] ) * hh[9] + 
                        ( dm[d] * n * (l^2 - 1.0) + m * dn[d] * (l^2 - 1.0) + m * n * 2*l * dl[d] ) * hh[10]
            dh[d,5,8] = 3.0/2 * l * m * (l^2 - m^2) * dhh[8] * dR[d] + 2.0 * l * m * (m^2 - l^2) * dhh[9] * dR[d] + 
						1.0/2 * l * m * (l^2 - m^2) * dhh[10] * dR[d] + 
                        3.0/2 * ( dl[d] * m * (l^2 - m^2) + l * dm[d] * (l^2 - m^2) + l * m * (2*l * dl[d] - 2*m * dm[d]) ) * hh[8] + 
                        2.0 * ( dl[d] * m * (m^2 - l^2) + l * dm[d] * (m^2 - l^2) + l * m * (2*m * dm[d] - 2*l * dl[d]) ) * hh[9] + 
                        ( dl[d] * m * (l^2 - m^2)/2 + l * dm[d] * (l^2 - m^2)/2 + l * m * (l*dl[d] - m*dm[d]) ) * hh[10]
            dh[d,5,9] = √3 * l * m * (n^2 - (l^2 + m^2)/2) * dhh[8] * dR[d] - 2.0*√3 * l * m * n^2 * dhh[9] * dR[d] + 
						√3/2 * l * m * (1.0 + n^2) * dhh[10] * dR[d] + 
                        √3 * ( dl[d] * m * (n^2 - (l^2 + m^2)/2) + l * dm[d] * (n^2 - (l^2 + m^2)/2) + l * m * (2*n * dn[d] - (l * dl[d] + m * dm[d])) ) * hh[8] - 
                        2.0*√3 * ( dl[d] * m * n^2 + l * dm[d] * n^2 + l * m * 2*n * dn[d] ) * hh[9] + 
                        √3/2 * ( dl[d] * m * (1.0 + n^2) + l * dm[d] * (1.0 + n^2) + l * m * 2*n * dn[d] ) * hh[10]
            dh[d,6,7] = 3.0 * l * m * n^2 * dhh[8] * dR[d] + l * m * (1.0 - 4.0 * n^2) * dhh[9] * dR[d] + 
						l * m * (n^2 - 1.0) * dhh[10] * dR[d] + 
                        3.0 * ( dl[d] * m * n^2 + l * dm[d] * n^2 + l * m * 2*n * dn[d] ) * hh[8] + 
                        ( dl[d] * m * (1.0 - 4.0 * n^2) + l * dm[d] * (1.0 - 4.0 * n^2) - l * m * 8*n * dn[d] ) * hh[9] + 
                        ( dl[d] * m * (n^2 - 1.0) + l * dm[d] * (n^2 - 1.0) + l * m * 2*n * dn[d] ) * hh[10]
            dh[d,6,8] = 3.0/2 * m * n * (l^2 - m^2) * dhh[8] * dR[d] - m * n * (1.0 + 2.0 * (l^2 - m^2)) * dhh[9] * dR[d] + 
						m * n * (1.0 + (l^2 - m^2) /2) * dhh[10] * dR[d] + 
                        3.0/2 * ( dm[d] * n * (l^2 - m^2) + m * dn[d] * (l^2 - m^2) + m * n * (2*l * dl[d] - 2*m * dm[d]) ) * hh[8] - 
                        ( dm[d] * n * (1.0 + 2.0 * (l^2 - m^2)) + m * dn[d] * (1.0 + 2.0 * (l^2 - m^2)) + m * n * 4.0 * (l * dl[d] - m * dm[d]) ) * hh[9] + 
                        ( dm[d] * n * (1.0 + (l^2 - m^2) /2) + m * dn[d] * (1.0 + (l^2 - m^2) /2) + m * n * (l * dl[d] - m * dm[d]) ) * hh[10]
            dh[d,6,9] = √3 * m * n * (n^2 - (l^2 + m^2) /2) * dhh[8] * dR[d] + √3 * m * n * (l^2 + m^2 - n^2) * dhh[9] * dR[d] - 
						√3/2 * m * n * (l^2 + m^2) * dhh[10] * dR[d] + 
                        √3 * ( dm[d] * n * (n^2 - (l^2 + m^2) /2) + m * dn[d] * (n^2 - (l^2 + m^2) /2) +  m * n * (2*n * dn[d] - (l * dl[d] + m * dm[d])) ) * hh[8] + 
                        √3 * ( dm[d] * n * (l^2 + m^2 - n^2) + m * dn[d] * (l^2 + m^2 - n^2) + m * n * (2*l * dl[d] + 2*m * dm[d] - 2*n * dn[d]) ) * hh[9] - 
                        √3/2 * ( dm[d] * n * (l^2 + m^2) + m * dn[d] * (l^2 + m^2) + m * n * (2*l * dl[d] + 2*m * dm[d]) ) * hh[10]
            dh[d,7,8] = 3.0/2 * l * n * (l^2 - m^2) * dhh[8] * dR[d] + l * n * (1.0 - 2.0 * (l^2 - m^2)) * dhh[9] * dR[d] - 
						l * n * (1.0 - (l^2 - m^2) /2) * dhh[10] * dR[d] + 
                        3.0/2 * ( dl[d] * n * (l^2 - m^2) + l * dn[d] * (l^2 - m^2) + l * n * (2*l * dl[d] - 2*m * dm[d]) ) * hh[8] + 
                        ( dl[d] * n * (1.0 - 2.0 * (l^2 - m^2)) + l * dn[d] * (1.0 - 2.0 * (l^2 - m^2)) - l * n * 4.0 * (l * dl[d] - m * dm[d]) ) * hh[9] - 
                        ( dl[d] * n * (1.0 - (l^2 - m^2) /2) + l * dn[d] * (1.0 - (l^2 - m^2) /2) - l * n * (l * dl[d] - m * dm[d]) ) * hh[10]
            dh[d,7,9] = √3 * l * n * (n^2 - (l^2 + m^2) /2) * dhh[8] * dR[d] + √3 * l * n * (l^2 + m^2 - n^2) * dhh[9] * dR[d] - 
						√3/2 * l * n * (l^2 + m^2) * dhh[10] * dR[d] + 
                        √3 * ( dl[d] * n * (n^2 - (l^2 + m^2) /2) + l * dn[d] * (n^2 - (l^2 + m^2) /2) + l * n * (2*n * dn[d] - (l * dl[d] + m * dm[d])) ) * hh[8] + 
                        √3 * ( dl[d] * n * (l^2 + m^2 - n^2) + l * dn[d] * (l^2 + m^2 - n^2) + l * n * (2*l * dl[d] + 2*m * dm[d] - 2*n * dn[d]) ) * hh[9] - 
                        √3/2 * ( dl[d] * n * (l^2 + m^2) + l * dn[d] * (l^2 + m^2) + l * n * (2*l * dl[d] + 2*m * dm[d]) ) * hh[10]
            dh[d,8,9] = √3/2 * (l^2 - m^2) * (n^2 - (l^2 + m^2) /2) * dhh[8] * dR[d] + √3 * n^2 * (m^2 - l^2) * dhh[9] * dR[d] + 
						√3/4 * (1.0 + n^2) * (l^2 - m^2) * dhh[10] * dR[d] + 
                        √3/2 * ( (2*l * dl[d] - 2*m * dm[d]) * (n^2 - (l^2 + m^2) /2) + (l^2 - m^2) * (2*n * dn[d] - (l * dl[d] + m * dm[d])) ) * hh[8] + 
                        √3 * ( 2*n * dn[d] * (m^2 - l^2) + n^2 * (2*m * dm[d] - 2*l * dl[d]) ) * hh[9] + 
                        √3/4 * ( 2*n * dn[d] * (l^2 - m^2) + (1.0 + n^2) * (2*l * dl[d] - 2*m * dm[d]) ) * hh[10]
            dh[d,6,5] = dh[d,5,6]
            dh[d,7,5] = dh[d,5,7]
            dh[d,8,5] = dh[d,5,8]
            dh[d,9,5] = dh[d,5,9]
            dh[d,7,6] = dh[d,6,7]
            dh[d,8,6] = dh[d,6,8]
            dh[d,9,6] = dh[d,6,9]
            dh[d,8,7] = dh[d,7,8]
            dh[d,9,7] = dh[d,7,9]
            dh[d,9,8] = dh[d,8,9]
        end

    else
        throw(ArgumentError("the numbers of atomic orbitals and bond types do not match!"))
    end
    return dh
end









############################### DATAS FOR NRL-TB #######################################


# CARBON
# 'C' : carbon with s&p orbitals
C_sp  =  NRLParams( 4, 4,				# norbital, nbond
                    10.5, 0.5,			# Rc, lc
                    1.59901905594,		# λ
                    [-0.102789972814  0.542619178314  0.542619178314  0.542619178314],  	#a
                    [-1.62604640052   2.73454062799   2.73454062799   2.73454062799],   	#b
                    [-178.884826119  -67.139709883   -67.139709883  -67.139709883],     	#c
                    [4516.11342028    438.52883145   438.52883145   438.52883145],      	#d
                    [74.0837449667   -7.9172955767   -5.7016933899   24.9104111573],  		#e
                    [-18.3225697598   3.6163510241   1.0450894823   -5.0603652530],   		#f
                    [-12.5253007169   1.0416715714   1.5062731505   -3.6844386855],   		#g
                    [1.41100521808   1.16878908431   1.13627440135   1.36548919302],  		#h
                    [0.18525064246   1.85250642463   -1.29666913067   0.74092406925],   	#p
                    [1.56010486948   -2.50183774417   0.28270660019   -0.07310263856],  	#q
                    [-0.308751658739   0.178540723033   -0.022234235553   0.016694077196],  #r
                    [1.13700564649   1.12900344616   0.76177690688   1.02148246334],  		#s
                   )


# ALUMINIUM
# 'Al' : Aluminum with s&p&d orbitals
Al_spd  =  NRLParams(9, 10,					# norbital, nbond
                     16.5, 0.5,				# Rc, lc
                     1.108515601511,		# λ
                     [-0.0368090795665,  0.394060871550,  0.394060871550,  0.394060871550,
                      1.03732517161,  1.03732517161,  1.03732517161, 1.03732517161, 1.03732517161],         	#a
                     [1.41121601477,  0.996479629379,   0.996479629379,  0.996479629379,
                      2.25910876474,  2.25910876474,  2.25910876474,  2.25910876474,  2.25910876474],       	#b
                     [13.7933378593,   7.02388797078,    7.02388797078,   7.02388797078,
                      -34.3716752929,  -34.3716752929,  -34.3716752929,  -34.3716752929,  -34.3716752929],  	#c
                     [-150.317796096,   -77.7996182049,  -77.7996182049,  -77.7996182049,
                      293.811629762,  293.811629762,  293.811629762,  293.811629762,  293.811629762],       	#d

                     [-45.1824404773,   11.1265443008,  -27.7049616756,   7.48992761254,  -29.0814831341,
                      0.843008479887,   35.6686973234,  -8939.68482560,   -55.7867097600,  41.7418125111],      #e
                     [19.0441568385,   -4.74833564810,   1.14888504976,   3.01675751048,   12.2929753319,
                      -1.52618018997,  -8.20900836372,   730.518353338,   0.853972348751,  -12.0915149851],     #f
                     [-2.81748968422,   0.273179395549,    1.33493438322,   -1.27543114992,   -1.75865704211,
                      0.378014000015,   -0.777295830901,   282.319390496,   2.30786449075,    0.905794614748],  #g
                     [1.05012810116,    0.880778921750,    0.983097680652,    1.01546352470,    1.03433851149,
                      0.964916606335,   1.08355725127,    1.35770591641,    0.997222907112,    0.898850379448], #h

                     [-20.9441522845,   -13.0267838833,   -13.7830192613,   560.641345191,   -103.077140819,
                      20.8403415336,   23.9108771944,   -295.728688028,   -17.2027621079,   -42.9299886946],    #p
                     [17.5240112799,   7.92017585690,   5.15785376743,   -215.309856616,   33.5869977281,
                      -6.65151229760,   -5.86527866863,   80.7470264719,   -6.54916621891,   23.2260645871],    #q
                     [-1.33002925172,   -0.523366384472,   -1.08061004452,   24.2658321255,   -1.86799882707,
                      0.195368101148,   0.725698443913,   -2.93711938026,   1.11096322734,   -0.538315401068],  #r
                     [1.06584516722,   0.943623371617,   0.915429215594,   1.17753799190,   0.988337965212,
                      0.873041790591,   0.999293973116,   1.02005972107,   1.01466433826,   1.14341718458],     #s
                    )


end

