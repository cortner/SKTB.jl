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
potential_energy_d, site_energy, band_structure_all, band_structure_near_eF


abstract AbstractTBModel <: AbstractCalculator
abstract SmearingFunction <: SimpleFunction









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




############################################################
### Site Energy Stuff


function site_energy(l::Integer, atm::ASEAtoms, tbm::TBModel)

    # tell tbm to update the spectral decompositions
    update!(atm, tbm)
    # BZ integration loop
    K, weight = monkhorstpackgrid(atm, tbm)

	# use the following parameters as those in update_eig!
	nlist = NeighbourList(cutoff(tbm), atm)
    nnz_est = length(nlist) * tbm.norbitals^2 + length(atm) * tbm.norbitals^2
    It = zeros(Int32, nnz_est)
    Jt = zeros(Int32, nnz_est)
    Ht = zeros(Complex{Float64}, nnz_est)
    Mt = zeros(Complex{Float64}, nnz_est)
    X = positions(atm)

    Es = 0.0
    for n = 1:size(K, 2)
        k = K[:, n]
        epsn = get_k_array(tbm, :epsn, k)
    	C = get_k_array(tbm, :C, k)::Matrix{Complex{Float64}}
	 	# precompute electron distribution function
		f = tbm.smearing(epsn, tbm.eF) .* epsn

    	# overlap matrix is needed in this calculation
	    # ([M^{1/2}*ψ]_i)^2 → [M*ψ]_i*[ψ]_i
        H, M = hamiltonian!(tbm, k, It, Jt, Ht, Mt, nlist, X)
		MC = M * C::Matrix{Complex{Float64}}

		I = indexblock(l, tbm)
		for j = 1:tbm.norbitals
			# the first component of the following line should be conjugate
			Es += weight[n] * r_sum(real(f .* slice(C, I[j], :) .* slice(MC, I[j], :)))
			# Es += weight[n] * r_sum( f .* (slice(C, I[j], :) .* slice(MC, I[j], :)) )
		end
	end

    return Es
end



site_energy(nn::Array{Int}, atm::ASEAtoms, tbm::TBModel) =
    reshape(Float64[ site_energy(n, atm, tbm) for n in nn ], size(nn))




# site_forces always returns a complete gradient, i.e. dEs = d x Natm
# When idx is an array, then the return-value is the gradient of \sum_{i ∈ idx} E_i

function site_forces(idx::Array{Int,1}, atm::ASEAtoms, tbm::TBModel)

    # tell tbm to update the spectral decompositions
    update!(atm, tbm)
    # BZ integration loop
    K, weight = monkhorstpackgrid(atm, tbm)

    # allocate output
    sfrc = zeros(Float64, 3, length(atm))

    # precompute neighbourlist
    nlist = NeighbourList(cutoff(tbm), atm)
	Nneig = 1
    for (n, neigs, r, R) in Sites(nlist)
        if length(neigs) > Nneig
            Nneig = length(neigs)
        end
    end

    X = positions(atm)
	# assemble the site forces for k-points
    for iK = 1:size(K,2)
        sfrc +=  weight[iK] *
            real(site_forces_k(idx, X, tbm, nlist, Nneig, K[:,iK]))
    end

    return sfrc
end



# scalar index: just wraps the vector version
site_forces(n::Int, atm::ASEAtoms, tbm::TBModel) = site_forces([n;], atm, tbm)



# With a given k-point, compute the site force by loop through eigenpairs (index s)
# note that in the old version, we loop through through atoms
#     E_l 	= ∑_s f(ɛ_s)⋅[ψ_s]_l^2
#     E_l 	= ∑_s f(ɛ_s)⋅[ψ_s]_l⋅[M⋅ψ_s]_l
# E_{l,n}	= ∑_s (	f'(ɛ_s)⋅ɛ_{s,n}⋅[ψ_s]_l⋅[M⋅ψ_s]_l + 2.0⋅f(ɛ_s)⋅[ψ_s]_{l,n}⋅[M⋅ψ_s]_l
#					+ f(ɛ_s)⋅[ψ_s]⋅[M_{,n}⋅ψ_s]_l )
# We loop through eigenpair s to compute the first two parts and through atom n for the third part
#
function site_forces_k(idx::Array{Int,1}, X::Matrix{Float64},
                       tbm::TBModel, nlist, Nneig, k::Vector{Float64};
                       beta = ones(size(X,2)))

    # obtain the precomputed arrays
    epsn = get_k_array(tbm, :epsn, k)
    C = get_k_array(tbm, :C, k)::Matrix{Complex{Float64}}
	# some constant parameters
    Nelc = length(epsn)
	Natm = size(X,2)
    Norb = tbm.norbitals

	# overlap matrix is needed in this calculation
	# use the following parameters as those in update_eig!
    nnz_est = length(nlist) * Norb^2 + Natm * Norb^2
    It = zeros(Int32, nnz_est)
    Jt = zeros(Int32, nnz_est)
    Ht = zeros(Complex{Float64}, nnz_est)
    Mt = zeros(Complex{Float64}, nnz_est)
    ~, M = hamiltonian!(tbm, k, It, Jt, Ht, Mt, nlist, X)
	MC = M * C::Matrix{Complex{Float64}}

    # allocate output
    const dEs = zeros(Complex{Float64}, 3, Natm)
    # pre-allocate dM
    const dM_nm = zeros(3, Norb, Norb)
    const Msn = zeros(Complex{Float64}, Nelc)
	# const eps_s_n = zeros(Float64, 3, Natm)
	# const psi_s_n = zeros(Float64, 3, Natm, Nelc)

	# precompute electron distribution function
	f = tbm.smearing(epsn, tbm.eF) .* epsn
    df = tbm.smearing(epsn, tbm.eF) + epsn .* (@D tbm.smearing(epsn, tbm.eF))

	# loop through all eigenstates to compute the hessian
	for s = 1 : Nelc
		# compute ϵ_{s,n} and ψ_{s,n}
		eps_s_n, psi_s_n = d_eigenstate_k(s, tbm, X, nlist, Nneig, k)

		# loop for the first part
 		for d = 1:3
 			for n = 1:Natm
				Msn = M * psi_s_n[d, n, :][:]
	            for id in idx
	                # in this iteration of the loop we compute the contributions
    	            # that come from the site i. hence multiply everything with beta[i]
        	        Ii = indexblock(id, tbm)
	 				dEs[d, n] -= beta[id] * df[s] * eps_s_n[d, n] * sum( C[Ii, s] .* MC[Ii, s] )
 					dEs[d, n] -= beta[id] * f[s] * sum( MC[Ii, s] .* psi_s_n[d, n, Ii][:] )
 					dEs[d, n] -= beta[id] * f[s] * sum( C[Ii, s] .* Msn[Ii] )
				end 	# loop of id
 			end		# loop of d
 		end		# loop of n
	end 	# loop of s

    # loop through all atoms, to compute the last part
    for (n, neigs, r, R) in Sites(nlist)
		# consider only the rows related to site idx
		if n in idx
	        # compute the block of indices for the orbitals belonging to n
    	    In = indexblock(n, tbm)
        	exp_i_kR = exp(im * (k' * (R - (X[:, neigs] .- X[:, n]))))

	        for i_n = 1:length(neigs)
    	      	m = neigs[i_n]
        	    Im = indexblock(m, tbm)
				eikr = exp_i_kR[i_n]

            	# compute and store ∂M_mn/∂y_n
	            grad!(tbm.overlap, r[i_n], R[:,i_n], dM_nm)
				# sum over all eigenpairs
		    	for s = 1:Nelc
    	    	    for d = 1:3
            	  		dEs[d, n] -= beta[n] * f[s] * sum( C[In, s] .* ( - slice(dM_nm, d, :, :) * C[Im, s] ) ) * eikr
              			dEs[d, m] -= beta[n] * f[s] * sum( C[In, s] .* ( slice(dM_nm, d, :, :) * C[Im, s] ) ) * eikr
		            end		# loop for d
				end 	# loop for s
			end		# loop for neighbour i_n
		end 	# end of if
	end		# loop for atom n

	return dEs
	# note that this is in fact the site force, -dEs
end





###################### Hessian and Higher-oerder derivatives ##########################



# For a given s and a given k-point, returns ψ_{s,n} and ϵ_{s,n} for all n∈{1,⋯,d×Natm}
# Input
#	 s : which eigenstate
#	 k : k-point
# Output
#	 psi_s_n : ψ_{s,n} for all n, a  3 × Natm × Nelc  matrix
#	 eps_s_n : ϵ_{s,n} for all n, a  3 × Natm         matrix
#
# Algorithm
#	 ϵ_{s,n} = < ψ_s | H_{,n} - ϵ_s * M,n | ψ_s >
#
#    ψ_{s,n} = ∑_{t,ϵ_t≠ϵ_s} ψ_t < ψ_t | ϵ_s⋅M_{,n} - H_{,n} | ψ_s > / (ϵ_t-ϵ_s)
#				- 1/2 ∑_{t,ϵ_t=ϵ_s} ψ_t < ψ_t | M_{,n} | ψ_s >
#
#    Step 1. compute  g_s_n = (ϵ_s⋅M_{,n} - H_{,n}) ⋅ ψ_s
#    		and  f_s_n = M_{,n} ⋅ ψ_s
#    Step 2. (C' * g) ./ (epsilon - epsilon[s])
# 			with the second part added in the loop for ϵ_t =≠ ϵ_s

function d_eigenstate_k(s::Int, tbm::TBModel, X::Matrix{Float64}, nlist, Nneig::Int,
						k::Vector{Float64})

	# obtain the precomputed arrays
    epsn = get_k_array(tbm, :epsn, k)
    C = get_k_array(tbm, :C, k)::Matrix{Complex{Float64}}

	# some constant parameters
    Nelc = length(epsn)
	Natm = size(X,2)
    Norb = tbm.norbitals

	# allocate memory
	psi_s_n = zeros(Complex{Float64}, 3*Natm, Nelc)
	eps_s_n = zeros(Complex{Float64}, 3*Natm)
	g_s_n = zeros(Complex{Float64}, 3*Natm, Nelc)
	f_s_n = zeros(Complex{Float64}, 3*Natm, Nelc)
	const dH_nn = zeros(3, Norb, Norb, Nneig)
    const dH_nm = zeros(3, Norb, Norb)
	const dM_nm = zeros(3, Norb, Norb)

	# Step 1. loop through all atoms to compute g_s_n and f_s_n for all n
    for (n, neigs, r, R) in Sites(nlist)
        In = indexblock(n, tbm)
        exp_i_kR = ( exp(im * (k' * (R - (X[:, neigs] .- X[:, n])))) )' # conjugate here for later use

        # compute and store ∂H_nn/∂y_n (onsite terms)
        evaluate_d!(tbm.onsite, r, R, dH_nn)

        for i_n = 1:length(neigs)
			m = neigs[i_n]
	        Im = indexblock(m, tbm)
			eikr = exp_i_kR[i_n]

            # compute and store ∂H_nm/∂y_m (hopping terms) and ∂M_nm/∂y_m
            grad!(tbm.hop, r[i_n], R[:,i_n], dH_nm)
            grad!(tbm.overlap, r[i_n], R[:,i_n], dM_nm)

			for d = 1:3
				md = d + 3*(m-1)
				nd = d + 3*(n-1)
				g_s_n[md, In] += ( slice(dH_nn, d, :, :, i_n) * C[In, s] )'
				g_s_n[nd, In] -= ( slice(dH_nn, d, :, :, i_n) * C[In, s] )'

                g_s_n[md, In] += ( slice(dH_nm, d, :, :) * C[Im, s] )' * eikr
       	        g_s_n[nd, In] -= ( slice(dH_nm, d, :, :) * C[Im, s] )' * eikr
                f_s_n[md, In] += ( slice(dM_nm, d, :, :) * C[Im, s] )' * eikr
   	            f_s_n[nd, In] -= ( slice(dM_nm, d, :, :) * C[Im, s] )' * eikr
			end		# loop for dimension

		end		# loop for neighbours
	end		# loop for atomic sites

	g_s_n = epsn[s] * f_s_n - g_s_n

	# Step 2. compute eps_s_n and psi_s_n for all n
	# TODO: use BLAS for matrix-matrix/vector multiplication?

	# compute ϵ_{s,n}
	eps_s_n = real( - g_s_n * C[:,s] )

	diff_eps_inv = zeros(Float64, Nelc)
	# loop through all orbitals to compute 1/(ϵ_t-ϵ_s) and add the second part of ψ_{s,n}
	for t = 1:Nelc
		if abs(epsn[t]-epsn[s]) > 1e-10
        	diff_eps_inv[t] = 1.0/(epsn[t]-epsn[s])
        else
        	diff_eps_inv[t] = 0.0
            # psi_s_n -= 0.5 * ( C[:,t] * (f_s_n * C[:,t])' )'
            psi_s_n -= 0.5 * transpose( C[:,t] * (f_s_n * C[:,t])' )
        end
	end 	# loop for orbitals

    # g = - (C' * gsn) ./ (epsilon - epsilon[s])
	# use BLAS here!! gemm!
	g_s_n = g_s_n * C
	for jj = 1 : Nelc
		@simd for ii = 1 : 3*Natm
			@inbounds g_s_n[ii,jj] *= diff_eps_inv[jj]
        end
    end
	# add the first part of ψ_{s,n}
	# psi_s_n += ( C * g_s_n' )'
	psi_s_n += transpose( C * g_s_n' )

	# return eps_s_n, psi_s_n
	return reshape(eps_s_n, 3, Natm), reshape(psi_s_n, 3, Natm, Nelc)
end




# hessian always returns a complete hessian, i.e. hessian = ( d × Natm )^2
function hessian(atm::ASEAtoms, tbm::TBModel)

    # tell tbm to update the spectral decompositions
    update!(atm, tbm)
    # BZ integration loop
    K, weight = monkhorstpackgrid(atm, tbm)
    # allocate output
    hessian = zeros(Float64, 3, length(atm), 3, length(atm))

    # precompute neighbourlist
    nlist = NeighbourList(cutoff(tbm), atm)
    Nneig = 1
    for (n, neigs, r, R) in Sites(nlist)
        if length(neigs) > Nneig
            Nneig = length(neigs)
        end
    end

    X = positions(atm)
    # loop for all k-points
    for iK = 1:size(K,2)
        Hess_k, ~ = hessian_k(X, tbm, nlist, Nneig, K[:,iK])
        hessian +=  weight[iK] * real(Hess_k)
    end

    return hessian
end



potential_energy_d2(atm::ASEAtoms, tbm::TBModel) = hessian(atm, tbm)



# Using 2n+1 theorem to compute hessian for a given k-point
# E_{,n}  =  ∑_s ( f(ϵ_s) + ϵ_s * f'(ϵ_s) ) * ϵ_{s,n}
# E_{,mn} =  ∑_s ( (2 * f'(ϵ_s) + ϵ_s * f''(ϵ_s) ) * ϵ_{s,m} * ϵ_{s,n}
#			 	   + ( f(ϵ_s) + ϵ_s * f'(ϵ_s) ) * ϵ_{s,mn} )
# with
# ϵ_{s,mn} = <ψ_s|H_{,mn}-ϵM_{,mn}-ϵ_{,n}M_{,m}-ϵ_{,m}M_{,n}|ψ_s>
#				 + <ψ_s|H_{,n}-ϵ_{,n}M-ϵM_{,n}|ψ_{s,m}>
#				 + <ψ_s|H_{,m}-ϵ_{,m}M-ϵM_{,m}|ψ_{s,n}>
#
# Output
# 		hessian ∈ R^{ 3 × Natm × 3 × Natm }
#       ɛ_{s,mn} ∈ R^{ Nelc ×  3 × Natm × 3 × Natm }
# note that the output of  ɛ_{s,mn}  is stored for usage of computing d3E
# TODO: have not added e^ikr into the hamiltonian yet
# TODO: we do not need the whole hessian matrix, but only those related to the centred atom

function hessian_k(X::Matrix{Float64}, tbm::TBModel, nlist, Nneig, k::Vector{Float64})

    # obtain the precomputed arrays
    epsn = get_k_array(tbm, :epsn, k)
    C = get_k_array(tbm, :C, k)::Matrix{Complex{Float64}}

	# some constant parameters
    Nelc = length(epsn)
	Natm = size(X,2)
    Norb = tbm.norbitals
	# "nlist" and "Nneig" from parameters
	eF = tbm.eF
	beta = tbm.smearing.beta

	# overlap matrix is needed in this calculation
	# use the following parameters as those in update_eig!
    nnz_est = length(nlist) * Norb^2 + Natm * Norb^2
    It = zeros(Int32, nnz_est)
    Jt = zeros(Int32, nnz_est)
    Ht = zeros(Complex{Float64}, nnz_est)
    Mt = zeros(Complex{Float64}, nnz_est)
    ~, M = hamiltonian!(tbm, k, It, Jt, Ht, Mt, nlist, X)
	MC = M * C::Matrix{Complex{Float64}}

    # allocate output
    eps_s_mn = zeros(Complex{Float64}, Nelc, 3, Natm, 3, Natm)
    Hess = zeros(Complex{Float64}, 3, Natm, 3, Natm)

    # pre-allocate dH, note that all of them will be computed by ForwardDiff
    # TODO: it seems much more convenient to evaluate the onsite Hamiltonians
	#		only the diagonal elements
	dH_nn  = zeros(3*Nneig, Norb)
    d2H_nn = zeros(3*Nneig, 3*Nneig, Norb)
    dH_nm  = zeros(3, Norb, Norb)
    d2H_nm = zeros(3, 3, Norb, Norb)
    M_nm   = zeros(Norb, Norb)
    dM_nm  = zeros(3, Norb, Norb)
    d2M_nm = zeros(3, 3, Norb, Norb)

	# const eps_s_n = zeros(Float64, 3, Natm)
	# const psi_s_n = zeros(Float64, 3, Natm, Nelc)

	# precompute electron distribution function
	# TODO: update potential.jl by adding @D2 and @D3 for smearing function
	feps1 = 2.0 * fermi_dirac_d(eF, beta, epsn) + epsn .* fermi_dirac_d2(eF, beta, epsn)
	feps2 = fermi_dirac(eF, beta, epsn) + epsn .* fermi_dirac_d(eF, beta, epsn)

	# loop through all eigenstates to compute the hessian
	for s = 1 : Nelc
		# compute ϵ_{s,n} and ψ_{s,n}
		eps_s_n, psi_s_n = d_eigenstate_k(s, tbm, X, nlist, Nneig, k)

		# loop for the first part
 		for d1 = 1:3
 			for n = 1:Natm
 				for d2 = 1:3
 					for m = 1:Natm
						# (2 * f'(ϵ_s) + ϵ_s * f''(ϵ_s) ) * ϵ_{s,m} * ϵ_{s,n}
 						Hess[d1, n, d2, m] += feps1[s] * eps_s_n[d1,n] * eps_s_n[d2,m]
						# and < ψ_s | -ϵ_{,n}M | ψ_{s,m} > + < ψ_s | -ϵ_{,m}M | ψ_{s,n} >
						# which is only 0 when the overlap matrix is identity matrix
                        eps_s_mn[s, d1, n, d2, m] += (
 			  			 	 - eps_s_n[d1, n] * MC[:, s]' * psi_s_n[d2, m, :][:]
	 						 - eps_s_n[d2, m] * MC[:, s]' * psi_s_n[d1, n, :][:]
 							 )[1]
 					end
 				end
 			end
 		end

	    # loop through all atoms for the second part, i.e. ϵ_{s,nm}
    	for (n, neigs, r, R) in Sites(nlist)
        	In = indexblock(n, tbm)
	        exp_i_kR = exp(im * (k' * (R - (X[:, neigs] .- X[:, n]))))

        	evaluate_fd!(tbm.onsite, R, dH_nn)
        	evaluate_fd2!(tbm.onsite, R, d2H_nn)

			# loop through all neighbours of the n-th site
    	    for i_n = 1:length(neigs)
				m = neigs[i_n]
		        Im = indexblock(m, tbm)
				eikr = exp_i_kR[i_n]

        	    # compute and store ∂H, ∂^2H and ∂M, ∂^2M
            	# evaluate!(tbm.overlap, r[i_n], R[:, i_n], M_nm)
            	evaluate_fd!(tbm.hop, R[:,i_n], dH_nm)
            	evaluate_fd2!(tbm.hop, R[:,i_n], d2H_nm)
        	    evaluate_fd!(tbm.overlap, R[:,i_n], dM_nm)
        	    evaluate_fd2!(tbm.overlap, R[:,i_n], d2M_nm)

				for d1 = 1:3
					for d2 = 1:3
						# contributions from hopping terms
						# from H_{nm,n} and H_{nm,m} to E_{,nk}, E_{,kn}, E_{,mk}, E_{,km}
						for l = 1 : Natm
							eps_s_mn[s, d1, n, d2, l] += (
								 C[In, s]' * ( - slice(dH_nm, d1, :, :)
								 + epsn[s] * slice(dM_nm, d1, :, :)
                               	              ) * psi_s_n[d2, l, Im][:]
								 + C[In, s]' * (
								 eps_s_n[d2, l] * slice(dM_nm, d1, :, :)
                                              ) * C[Im,s]
								 )[1] * eikr
							eps_s_mn[s, d1, l, d2, n] += (
								 C[In, s]' * ( - slice(dH_nm, d2, :, :)
								 + epsn[s] * slice(dM_nm, d2, :, :)
                               	              ) * psi_s_n[d1, l, Im][:]
								 + C[In, s]' * (
								 eps_s_n[d1, l] * slice(dM_nm, d2, :, :)
                                              ) * C[Im,s]
								 )[1] * eikr
							eps_s_mn[s, d1, m, d2, l] += (
								 C[In, s]' * ( slice(dH_nm, d1, :, :)
								 - epsn[s] * slice(dM_nm, d1, :, :)
                               	              ) * psi_s_n[d2, l, Im][:]
								 - C[In, s]' * (
								 eps_s_n[d2, l] * slice(dM_nm, d1, :, :)
                                              ) * C[Im,s]
								 )[1] * eikr
							eps_s_mn[s, d1, l, d2, m] += (
								 C[In, s]' * ( slice(dH_nm, d2, :, :)
								 - epsn[s] * slice(dM_nm, d2, :, :)
                               	              ) * psi_s_n[d1, l, Im][:]
								 - C[In, s]' * (
								 eps_s_n[d1, l] * slice(dM_nm, d2, :, :)
                                              ) * C[Im,s]
								 )[1] * eikr
						end	# loop for atom l

						# contributions from hopping terms
						# 4 parts: from H_{nm,nn}, H_{nm,mm}, H_{nm,mn}, H_{nm,nm}
						eps_s_mn[s, d1, n, d2, n] += (
								 C[In, s]' * ( slice(d2H_nm, d1, d2, :, :)
								 - epsn[s] * slice(d2M_nm, d1, d2, :, :) ) * C[Im,s]
								 )[1] * eikr
						eps_s_mn[s, d1, m, d2, m] += (
								 C[In, s]' * ( slice(d2H_nm, d1, d2, :, :)
								 - epsn[s] * slice(d2M_nm, d1, d2, :, :) ) * C[Im,s]
								 )[1] * eikr
						eps_s_mn[s, d1, m, d2, n] += (
								 C[In, s]' * ( - slice(d2H_nm, d1, d2, :, :)
								 + epsn[s] * slice(d2M_nm, d1, d2, :, :) ) * C[Im,s]
								 )[1] * eikr
						eps_s_mn[s, d1, n, d2, m] += (
								 C[In, s]' * ( - slice(d2H_nm, d1, d2, :, :)
								 + epsn[s] * slice(d2M_nm, d1, d2, :, :) ) * C[Im,s]
								 )[1] * eikr

						# contributions from onsite terms
						m1 = 3*(i_n-1) + d1
						m2 = 3*(i_n-1) + d2

						# from H_{nn,n} and H_{nn,m} to E_{,nk}, E_{,kn}, E_{,mk}, E_{,km}
						for l = 1 : Natm
							eps_s_mn[s, d1, n, d2, l] += (
								 C[In, s]' * ( - slice(dH_nn, m1, :) .* psi_s_n[d2, l, In][:] )
								 )[1]
							eps_s_mn[s, d1, l, d2, n] += (
								 C[In, s]' * ( - slice(dH_nn, m2, :) .* psi_s_n[d1, l, In][:] )
								 )[1]
							eps_s_mn[s, d1, m, d2, l] += (
								 C[In, s]' * ( slice(dH_nn, m1, :) .* psi_s_n[d2, l, In][:] )
								 )[1]
							eps_s_mn[s, d1, l, d2, m] += (
								 C[In, s]' * ( slice(dH_nn, m2, :) .* psi_s_n[d1, l, In][:] )
								 )[1]
						end	# loop for atom l

						# another loop for neighbours
						# 4 parts: from H_{nn,nn}, H_{nn,mm'}, H_{nn,mn}, H_{nn,nm}
						for i_m = 1:length(neigs)
							mm = neigs[i_m]
		    			    Imm = indexblock(mm, tbm)
							mm1 = 3*(i_m-1) + d1
							mm2 = 3*(i_m-1) + d2
 							eps_s_mn[s, d1, m, d2, mm] += (
 									 C[In, s]' * ( d2H_nn[m1, mm2, :][:] .* C[In,s] )
 									 )[1]
 							eps_s_mn[s, d1, m, d2, n] += (
 									 C[In, s]' * ( - d2H_nn[m1, mm2, :][:] .* C[In,s] )
 									 )[1]
 							eps_s_mn[s, d1, n, d2, m] += (
 									 C[In, s]' * ( - d2H_nn[mm1, m2, :][:] .* C[In,s] )
 									 )[1]
 							eps_s_mn[s, d1, n, d2, n] += (
									 C[In, s]' * ( d2H_nn[m1, mm2, :][:] .* C[In,s] )
									 )[1]
						end		# loop for neighbours i_m

					end		# loop for d2
				end		# loop for d1

			end		# loop for neighbours i_n
    	end		# loop for atomic sites

		# add eps_{s,mn} into the Hessian
 		for d1 = 1:3
 			for n = 1:Natm
 				for d2 = 1:3
 					for m = 1:Natm
						Hess[d1, m, d2, n] += feps2[s] * eps_s_mn[s, d1, m, d2, n]
					end
				end
			end
		end

    end		# loop for eigenstates

    return Hess, eps_s_mn
end




# d3E always returns a complete 3rd-order tensor, i.e. d3E = ( d × Natm )^3
function d3E(atm::ASEAtoms, tbm::TBModel)

    # tell tbm to update the spectral decompositions
    update!(atm, tbm)
    # BZ integration loop
    K, weight = monkhorstpackgrid(atm, tbm)
    # allocate output
    D3E = zeros(Float64, 3, length(atm), 3, length(atm), 3, length(atm))

    # precompute neighbourlist
    nlist = NeighbourList(cutoff(tbm), atm)
    Nneig = 1
    for (n, neigs, r, R) in Sites(nlist)
        if length(neigs) > Nneig
            Nneig = length(neigs)
        end
    end

    X = positions(atm)
    # loop for all k-points
    for iK = 1:size(K,2)
        D3E +=  weight[iK] * real(d3E_k(X, tbm, nlist, Nneig, K[:,iK]))
    end

    return D3E
end


potential_energy_d3(atm::ASEAtoms, tbm::TBModel) = d3E(atm, tbm)



# Using 2n+1 theorem to compute hessian for a given k-point
# E_{,lmn} =  ∑_s ( ( 3 * f''(ϵ_s) + ϵ_s * f'''(ϵ_s) ) * ϵ_{s,l} * ϵ_{s,m} * ϵ_{s,n}
#			 	  + ( 2 * f'(ϵ_s) + ϵ_s * f''(ϵ_s) ) * ( ϵ_{s,mn} * ϵ_{s,l}
#				  + ϵ_{s,lm} * ϵ_{s,n} + ϵ_{s,ln} * ϵ_{s,m} )
#			 	  + ( f(ϵ_s) + ϵ_s * f'(ϵ_s) ) * ϵ_{s,lmn} )
# with
# ϵ_{s,i} and ϵ_{s,jk} passed from previous calculations and
#
# ϵ_{s,ijk} = <ψ_s|H_{,ijk}-ϵM_{,ijk}-ϵ_{,i}M_{,jk}-ϵ_{,j}M_{,ik}-ϵ_{,k}M_{,ij}
#					-ϵ_{,ij}M_{,k}-ϵ_{,ik}M_{,j}-ϵ_{,jk}M_{,i}|ψ_s>
#			 + 2Re<ψ_s|H_{,jk}-ϵ_{,jk}M-ϵ_{,j}M_{,k}-ϵ_{,k}M_{,j}-ϵM_{,jk}|ψ_{s,i}>
#			 + 2Re<ψ_s|H_{,ik}-ϵ_{,ik}M-ϵ_{,i}M_{,k}-ϵ_{,k}M_{,i}-ϵM_{,ik}|ψ_{s,j}>
#			 + 2Re<ψ_s|H_{,ij}-ϵ_{,ij}M-ϵ_{,i}M_{,j}-ϵ_{,j}M_{,i}-ϵM_{,ij}|ψ_{s,k}>
#			 + 2Re<ψ_{s,i}|H_{,k}-ϵ_{,k}M-ϵM_{,k}|ψ_{s,j}>
#			 + 2Re<ψ_{s,i}|H_{,j}-ϵ_{,j}M-ϵM_{,j}|ψ_{s,k}>
#			 + 2Re<ψ_{s,j}|H_{,i}-ϵ_{,i}M-ϵM_{,i}|ψ_{s,k}>
#
# Output
# 		d3E ∈ R^{ 3 × Natm × 3 × Natm × 3 × Natm }
# TODO: have not added e^ikr into the hamiltonian yet

function d3E_k(X::Matrix{Float64}, tbm::TBModel, nlist, Nneig, k::Vector{Float64})

    # obtain the precomputed arrays
    epsn = get_k_array(tbm, :epsn, k)
    C = get_k_array(tbm, :C, k)::Matrix{Complex{Float64}}

	# some constant parameters
    Nelc = length(epsn)
	Natm = size(X,2)
    Norb = tbm.norbitals
	# "nlist" and "Nneig" from parameters
	eF = tbm.eF
	beta = tbm.smearing.beta

	# overlap matrix is needed in this calculation
	# use the following parameters as those in update_eig!
    nnz_est = length(nlist) * Norb^2 + Natm * Norb^2
    It = zeros(Int32, nnz_est)
    Jt = zeros(Int32, nnz_est)
    Ht = zeros(Complex{Float64}, nnz_est)
    Mt = zeros(Complex{Float64}, nnz_est)
    ~, M = hamiltonian!(tbm, k, It, Jt, Ht, Mt, nlist, X)
	MC = M * C::Matrix{Complex{Float64}}

    # allocate output
    const D3E = zeros(Complex{Float64}, 3, Natm, 3, Natm, 3, Natm)

    # pre-allocate dH, note that all of them will be computed by ForwardDiff
	dH_nn  = zeros(3*Nneig, Norb)
    d2H_nn = zeros(3*Nneig, 3*Nneig, Norb)
    d3H_nn = zeros(3*Nneig, 3*Nneig, 3*Nneig, Norb)
    dH_nm  = zeros(3, Norb, Norb)
    d2H_nm = zeros(3, 3, Norb, Norb)
    d3H_nm = zeros(3, 3, 3, Norb, Norb)
    M_nm   = zeros(Norb, Norb)
    dM_nm  = zeros(3, Norb, Norb)
    d2M_nm = zeros(3, 3, Norb, Norb)
    d3M_nm = zeros(3, 3, 3, Norb, Norb)

	# const eps_s_n = zeros(Float64, 3, Natm)
	# const psi_s_n = zeros(Float64, 3, Natm, Nelc)
	# const eps_s_mn = zeros(Float64, 3, Natm, 3, Natm)

	# precompute the 2nd order derivatives of the eigenvalues, ɛ_{s,mn}
	~, eps_s_mn = hessian_k(X, tbm, nlist, Nneig, k)

	# precompute electron distribution function
	# TODO: update potential.jl by adding @D2 and @D3 for smearing function
	feps1 = 3.0 * fermi_dirac_d2(eF, beta, epsn) + epsn .* fermi_dirac_d3(eF, beta, epsn)
	feps2 = 2.0 * fermi_dirac_d(eF, beta, epsn) + epsn .* fermi_dirac_d2(eF, beta, epsn)
	feps3 = fermi_dirac(eF, beta, epsn) + epsn .* fermi_dirac_d(eF, beta, epsn)

	# loop through all eigenstates to compute the hessian
	for s = 1 : Nelc
		# compute ϵ_{s,n} and ψ_{s,n}
		eps_s_n, psi_s_n = d_eigenstate_k(s, tbm, X, nlist, Nneig, k)

		# loop for the first part  ϵ_{s,l} * ϵ_{s,m} * ϵ_{s,n}
		# and second part  ϵ_{s,mn} * ϵ_{s,l} + ϵ_{s,lm} * ϵ_{s,n} + ϵ_{s,ln} * ϵ_{s,m}
		for d1 = 1:3
			for l = 1:Natm
				for d2 = 1:3
					for m = 1:Natm
						for d3 = 1:3
							for n = 1:Natm
								D3E[d1, l, d2, m, d3, n] +=
									feps1[s] * eps_s_n[d1,l] * eps_s_n[d2,m] * eps_s_n[d3,n] + feps2[s] *
									( eps_s_mn[s, d1, l, d2, m] * eps_s_n[d3, n]
									+ eps_s_mn[s, d2, m, d3, n] * eps_s_n[d1, l]
									+ eps_s_mn[s, d3, n, d1, l] * eps_s_n[d2, m] )
								# and all the terms with overlap matrix M
								# which is only 0 when the overlap matrix is identity matrix
								D3E[d1, l, d2, m, d3, n] += 2.0 * feps3[s] * (
			 						- eps_s_mn[s, d2, m, d3, n] * MC[:, s]' * psi_s_n[d1, l, :][:]
 						 			- eps_s_mn[s, d1, l, d2, m] * MC[:, s]' * psi_s_n[d3, n, :][:]
 						 			- eps_s_mn[s, d1, l, d3, n] * MC[:, s]' * psi_s_n[d2, m, :][:]
									- eps_s_n[d1, l] * psi_s_n[d2, m, :][:]' * M * psi_s_n[d3, n, :][:]
									- eps_s_n[d2, m] * psi_s_n[d1, l, :][:]' * M * psi_s_n[d3, n, :][:]
									- eps_s_n[d3, n] * psi_s_n[d1, l, :][:]' * M * psi_s_n[d2, m, :][:]
									)[1]
							end
						end
					end
				end
			end
		end

	    # loop through all atoms for the second part, i.e. ϵ_{s,lmn}
    	for (n, neigs, r, R) in Sites(nlist)
        	In = indexblock(n, tbm)
	        exp_i_kR = exp(im * (k' * (R - (X[:, neigs] .- X[:, n]))))

        	evaluate_fd!(tbm.onsite, R, dH_nn)
        	evaluate_fd2!(tbm.onsite, R, d2H_nn)
        	evaluate_fd3!(tbm.onsite, R, d3H_nn)

			# loop through all neighbours of the n-th site
    	    for i_n = 1:length(neigs)
				m = neigs[i_n]
		        Im = indexblock(m, tbm)
				eikr = exp_i_kR[i_n]

        	    # compute and store ∂H, ∂^2H and ∂M, ∂^2M
            	evaluate_fd!(tbm.hop, R[:,i_n], dH_nm)
            	evaluate_fd2!(tbm.hop, R[:,i_n], d2H_nm)
            	evaluate_fd3!(tbm.hop, R[:,i_n], d3H_nm)
        	    evaluate_fd!(tbm.overlap, R[:,i_n], dM_nm)
        	    evaluate_fd2!(tbm.overlap, R[:,i_n], d2M_nm)
        	    evaluate_fd3!(tbm.overlap, R[:,i_n], d3M_nm)

				for d1 = 1:3
					for d2 = 1:3
						for d3 = 1:3
							# contributions from hopping terms
							# loop for all terms related to H_{,i} and M_{,i} where i can only be n or m
							for p = 1 : Natm
								for q = 1 : Natm
									# 1. npq
									D3E[d1, n, d2, p, d3, q] +=  feps3[s] * (
											eps_s_mn[s, d2, p, d3, q] * C[In, s]' * slice(dM_nm, d1, :, :) * C[Im, s]
											+ 2.0 * eps_s_n[d2, p] * C[In, s]' * slice(dM_nm, d1, :, :) * psi_s_n[d3, q, Im][:]
											+ 2.0 * eps_s_n[d3, q] * C[In, s]' * slice(dM_nm, d1, :, :) * psi_s_n[d2, p, Im][:]
											+ 2.0 * psi_s_n[d2, p, In][:]' * ( - slice(dH_nm, d1, :, :)
								 			+ epsn[s] * slice(dM_nm, d1, :, :) ) * psi_s_n[d3, q, Im][:]
											)[1] * eikr
									# 2. mpq
									D3E[d1, m, d2, p, d3, q] +=  feps3[s] * (
											- eps_s_mn[s, d2, p, d3, q] * C[In, s]' * slice(dM_nm, d1, :, :) * C[Im, s]
											- 2.0 * eps_s_n[d2, p] * C[In, s]' * slice(dM_nm, d1, :, :) * psi_s_n[d3, q, Im][:]
											- 2.0 * eps_s_n[d3, q] * C[In, s]' * slice(dM_nm, d1, :, :) * psi_s_n[d2, p, Im][:]
											+ 2.0 * psi_s_n[d2, p, In][:]' * ( slice(dH_nm, d1, :, :)
								 			- epsn[s] * slice(dM_nm, d1, :, :) ) * psi_s_n[d3, q, Im][:]
											)[1] * eikr
									# 3. pnq
									D3E[d1, p, d2, n, d3, q] +=  feps3[s] * (
											eps_s_mn[s, d1, p, d3, q] * C[In, s]' * slice(dM_nm, d2, :, :) * C[Im, s]
											+ 2.0 * eps_s_n[d1, p] * C[In, s]' * slice(dM_nm, d2, :, :) * psi_s_n[d3, q, Im][:]
											+ 2.0 * eps_s_n[d3, q] * C[In, s]' * slice(dM_nm, d2, :, :) * psi_s_n[d1, p, Im][:]
											+ 2.0 * psi_s_n[d1, p, In][:]' * ( - slice(dH_nm, d2, :, :)
								 			+ epsn[s] * slice(dM_nm, d2, :, :) ) * psi_s_n[d3, q, Im][:]
											)[1] * eikr
									# 4. pmq
									D3E[d1, p, d2, m, d3, q] +=  feps3[s] * (
											- eps_s_mn[s, d1, p, d3, q] * C[In, s]' * slice(dM_nm, d2, :, :) * C[Im, s]
											- 2.0 * eps_s_n[d1, p] * C[In, s]' * slice(dM_nm, d2, :, :) * psi_s_n[d3, q, Im][:]
											- 2.0 * eps_s_n[d3, q] * C[In, s]' * slice(dM_nm, d2, :, :) * psi_s_n[d1, p, Im][:]
											+ 2.0 * psi_s_n[d1, p, In][:]' * ( slice(dH_nm, d2, :, :)
								 			- epsn[s] * slice(dM_nm, d2, :, :) ) * psi_s_n[d3, q, Im][:]
											)[1] * eikr
									# 5. pqn
									D3E[d1, p, d2, q, d3, n] +=  feps3[s] * (
											eps_s_mn[s, d1, p, d2, q] * C[In, s]' * slice(dM_nm, d3, :, :) * C[Im, s]
											+ 2.0 * eps_s_n[d1, p] * C[In, s]' * slice(dM_nm, d3, :, :) * psi_s_n[d2, q, Im][:]
											+ 2.0 * eps_s_n[d2, q] * C[In, s]' * slice(dM_nm, d3, :, :) * psi_s_n[d1, p, Im][:]
											+ 2.0 * psi_s_n[d1, p, In][:]' * ( - slice(dH_nm, d3, :, :)
								 			+ epsn[s] * slice(dM_nm, d3, :, :) ) * psi_s_n[d2, q, Im][:]
											)[1] * eikr
									# 6. pqm
									D3E[d1, p, d2, q, d3, m] +=  feps3[s] * (
											- eps_s_mn[s, d1, p, d2, q] * C[In, s]' * slice(dM_nm, d3, :, :) * C[Im, s]
											- 2.0 * eps_s_n[d1, p] * C[In, s]' * slice(dM_nm, d3, :, :) * psi_s_n[d2, q, Im][:]
											- 2.0 * eps_s_n[d2, q] * C[In, s]' * slice(dM_nm, d3, :, :) * psi_s_n[d1, p, Im][:]
											+ 2.0 * psi_s_n[d1, p, In][:]' * ( slice(dH_nm, d3, :, :)
								 			- epsn[s] * slice(dM_nm, d3, :, :) ) * psi_s_n[d2, q, Im][:]
											)[1] * eikr
								end 	# loop for atom p
							end 	# loop for atom q

							# contributions from hopping terms
							# loop for all terms related to H_{,ij} and M_{,ij}
							# where ij can only be nn, mm, nm, mn
							for l = 1 : Natm
								# 1. nnl
								D3E[d1, n, d2, n, d3, l] +=  feps3[s] * (
										- eps_s_n[d3, l] * C[In, s]' * slice(d2M_nm, d1, d2, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( slice(d2H_nm, d1, d2, :, :)
										- epsn[s] * slice(d2M_nm, d1, d2, :, :) ) * psi_s_n[d3, l, Im][:]
										)[1] * eikr
								# 2. mml
								D3E[d1, m, d2, m, d3, l] +=  feps3[s] * (
										- eps_s_n[d3, l] * C[In, s]' * slice(d2M_nm, d1, d2, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( slice(d2H_nm, d1, d2, :, :)
										- epsn[s] * slice(d2M_nm, d1, d2, :, :) ) * psi_s_n[d3, l, Im][:]
										)[1] * eikr
								# 3. nml
								D3E[d1, n, d2, m, d3, l] +=  feps3[s] * (
										eps_s_n[d3, l] * C[In, s]' * slice(d2M_nm, d1, d2, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( - slice(d2H_nm, d1, d2, :, :)
										+ epsn[s] * slice(d2M_nm, d1, d2, :, :) ) * psi_s_n[d3, l, Im][:]
										)[1] * eikr
								# 4. mnl
								D3E[d1, m, d2, n, d3, l] +=  feps3[s] * (
										eps_s_n[d3, l] * C[In, s]' * slice(d2M_nm, d1, d2, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( - slice(d2H_nm, d1, d2, :, :)
										+ epsn[s] * slice(d2M_nm, d1, d2, :, :) ) * psi_s_n[d3, l, Im][:]
										)[1] * eikr
								# 5. nln
								D3E[d1, n, d2, l, d3, n] +=  feps3[s] * (
										- eps_s_n[d2, l] * C[In, s]' * slice(d2M_nm, d1, d3, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( slice(d2H_nm, d1, d3, :, :)
										- epsn[s] * slice(d2M_nm, d1, d3, :, :) ) * psi_s_n[d2, l, Im][:]
										)[1] * eikr
								# 6. mlm
								D3E[d1, m, d2, l, d3, m] +=  feps3[s] * (
										- eps_s_n[d2, l] * C[In, s]' * slice(d2M_nm, d1, d3, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( slice(d2H_nm, d1, d3, :, :)
										- epsn[s] * slice(d2M_nm, d1, d3, :, :) ) * psi_s_n[d2, l, Im][:]
										)[1] * eikr
								# 7. nlm
								D3E[d1, n, d2, l, d3, m] +=  feps3[s] * (
										eps_s_n[d2, l] * C[In, s]' * slice(d2M_nm, d1, d3, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( - slice(d2H_nm, d1, d3, :, :)
										+ epsn[s] * slice(d2M_nm, d1, d3, :, :) ) * psi_s_n[d2, l, Im][:]
										)[1] * eikr
								# 8. mln
								D3E[d1, m, d2, l, d3, n] +=  feps3[s] * (
										eps_s_n[d2, l] * C[In, s]' * slice(d2M_nm, d1, d3, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( - slice(d2H_nm, d1, d3, :, :)
										+ epsn[s] * slice(d2M_nm, d1, d3, :, :) ) * psi_s_n[d2, l, Im][:]
										)[1] * eikr
								# 9. lnn
								D3E[d1, l, d2, n, d3, n] +=  feps3[s] * (
										- eps_s_n[d1, l] * C[In, s]' * slice(d2M_nm, d2, d3, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( slice(d2H_nm, d2, d3, :, :)
										- epsn[s] * slice(d2M_nm, d2, d3, :, :) ) * psi_s_n[d1, l, Im][:]
										)[1] * eikr
								# 10. lmm
								D3E[d1, l, d2, m, d3, m] +=  feps3[s] * (
										- eps_s_n[d1, l] * C[In, s]' * slice(d2M_nm, d2, d3, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( slice(d2H_nm, d2, d3, :, :)
										- epsn[s] * slice(d2M_nm, d2, d3, :, :) ) * psi_s_n[d1, l, Im][:]
										)[1] * eikr
								# 11. lnm
								D3E[d1, l, d2, n, d3, m] +=  feps3[s] * (
										eps_s_n[d1, l] * C[In, s]' * slice(d2M_nm, d2, d3, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( - slice(d2H_nm, d2, d3, :, :)
										+ epsn[s] * slice(d2M_nm, d2, d3, :, :) ) * psi_s_n[d1, l, Im][:]
										)[1] * eikr
								# 12. lmn
								D3E[d1, l, d2, m, d3, n] +=  feps3[s] * (
										eps_s_n[d1, l] * C[In, s]' * slice(d2M_nm, d2, d3, :, :) * C[Im, s]
										+ 2.0 * C[In, s]' * ( - slice(d2H_nm, d2, d3, :, :)
										+ epsn[s] * slice(d2M_nm, d2, d3, :, :) ) * psi_s_n[d1, l, Im][:]
										)[1] * eikr
							end 	# loop for atom l

							# contributions from hopping terms
							# loop for all terms related to  H_{nm,ijk} and M_{nm,ijk}
							# where ijk can only be  nnn, nnm, nmn, nmm, mmm, mmn, mnm, mnn
							# 1. nnn
							D3E[d1, n, d2, n, d3, n] +=  feps3[s] * (
								 	C[In, s]' * ( - slice(d3H_nm, d1, d2, d3, :, :)
									+ epsn[s] * slice(d3M_nm, d1, d2, d3, :, :)	) * C[Im, s]
									)[1] * eikr
							# 2. nnm
							D3E[d1, n, d2, n, d3, m] +=  feps3[s] * (
								 	C[In, s]' * ( slice(d3H_nm, d1, d2, d3, :, :)
									- epsn[s] * slice(d3M_nm, d1, d2, d3, :, :)	) * C[Im, s]
								 	)[1] * eikr
							# 3. nmn
							D3E[d1, n, d2, m, d3, n] +=  feps3[s] * (
								 	C[In, s]' * ( slice(d3H_nm, d1, d2, d3, :, :)
									- epsn[s] * slice(d3M_nm, d1, d2, d3, :, :)	) * C[Im, s]
									)[1] * eikr
							# 4. nmm
							D3E[d1, n, d2, m, d3, m] +=  feps3[s] * (
								 	C[In, s]' * ( - slice(d3H_nm, d1, d2, d3, :, :)
									+ epsn[s] * slice(d3M_nm, d1, d2, d3, :, :)	) * C[Im, s]
									)[1] * eikr
							# 5. mmm
							D3E[d1, m, d2, m, d3, m] +=  feps3[s] * (
								 	C[In, s]' * ( slice(d3H_nm, d1, d2, d3, :, :)
									- epsn[s] * slice(d3M_nm, d1, d2, d3, :, :)	) * C[Im, s]
									)[1] * eikr
							# 6. mmn
							D3E[d1, m, d2, m, d3, n] +=  feps3[s] * (
								 	C[In, s]' * ( - slice(d3H_nm, d1, d2, d3, :, :)
									+ epsn[s] * slice(d3M_nm, d1, d2, d3, :, :)	) * C[Im, s]
									)[1] * eikr
							# 7. mnm
							D3E[d1, m, d2, n, d3, m] +=  feps3[s] * (
								 	C[In, s]' * ( - slice(d3H_nm, d1, d2, d3, :, :)
									+ epsn[s] * slice(d3M_nm, d1, d2, d3, :, :)	) * C[Im, s]
									)[1] * eikr
							# 8. mnn
							D3E[d1, m, d2, n, d3, n] +=  feps3[s] * (
								 	C[In, s]' * ( slice(d3H_nm, d1, d2, d3, :, :)
									- epsn[s] * slice(d3M_nm, d1, d2, d3, :, :)	) * C[Im, s]
									)[1] * eikr


							# contributions from onsite terms
							m1 = 3*(i_n-1) + d1
							m2 = 3*(i_n-1) + d2
							m3 = 3*(i_n-1) + d3

							# loop for all terms related to H_{nn,i}
							# 6 parts:  where i can only be n or m
							for p = 1 : Natm
								for q = 1 : Natm
									# npq, mpq
									D3E[d1, n, d2, p, d3, q] +=  feps3[s] * (
 										2.0 * psi_s_n[d2, p, In][:]' * ( - dH_nn[m1, :][:] .* psi_s_n[d3, q, In][:] )
										)[1]
									D3E[d1, m, d2, p, d3, q] +=  feps3[s] * (
 										2.0 * psi_s_n[d2, p, In][:]' * ( dH_nn[m1, :][:] .* psi_s_n[d3, q, In][:] )
										)[1]
									# pnq, pmq
									D3E[d1, p, d2, n, d3, q] +=  feps3[s] * (
 										2.0 * psi_s_n[d1, p, In][:]' * ( - dH_nn[m2, :][:] .* psi_s_n[d3, q, In][:] )
										)[1]
									D3E[d1, p, d2, m, d3, q] +=  feps3[s] * (
 										2.0 * psi_s_n[d1, p, In][:]' * ( dH_nn[m2, :][:] .* psi_s_n[d3, q, In][:] )
										)[1]
									# pqn, pqm
									D3E[d1, p, d2, q, d3, n] +=  feps3[s] * (
 										2.0 * psi_s_n[d1, p, In][:]' * ( - dH_nn[m3, :][:] .* psi_s_n[d2, q, In][:] )
										)[1]
									D3E[d1, p, d2, q, d3, m] +=  feps3[s] * (
 										2.0 * psi_s_n[d1, p, In][:]' * ( dH_nn[m3, :][:] .* psi_s_n[d2, q, In][:] )
										)[1]
								end 	# loop for atom q
							end 	# loop for atom p

							# another loop for neighbors
							for i_m = 1:length(neigs)
								mm = neigs[i_m]
			    			    Imm = indexblock(mm, tbm)
								mm1 = 3*(i_m-1) + d1
								mm2 = 3*(i_m-1) + d2
								mm3 = 3*(i_m-1) + d3

								# loop for all terms related to H_{nn,ij}
								# 12 parts:  where ij can only be nn, mm, nm, mn
								for l = 1 : Natm
									# nnl, nml, mnl, mml
									D3E[d1, n, d2, n, d3, l] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( d2H_nn[m1, mm2, :][:] .* psi_s_n[d3, l, In][:] )
										)[1]
									D3E[d1, n, d2, m, d3, l] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( - d2H_nn[mm1, m2, :][:] .* psi_s_n[d3, l, In][:] )
										)[1]
									D3E[d1, m, d2, n, d3, l] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( - d2H_nn[m1, mm2, :][:] .* psi_s_n[d3, l, In][:] )
										)[1]
									D3E[d1, m, d2, mm, d3, l] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( d2H_nn[m1, mm2, :][:] .* psi_s_n[d3, l, In][:] )
										)[1]
									# nln, nlm, mln, mlm
									D3E[d1, n, d2, l, d3, n] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( d2H_nn[m1, mm3, :][:] .* psi_s_n[d2, l, In][:] )
										)[1]
									D3E[d1, n, d2, l, d3, m] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( - d2H_nn[mm1, m3, :][:] .* psi_s_n[d2, l, In][:] )
										)[1]
									D3E[d1, m, d2, l, d3, n] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( - d2H_nn[m1, mm3, :][:] .* psi_s_n[d2, l, In][:] )
										)[1]
									D3E[d1, m, d2, l, d3, mm] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( d2H_nn[m1, mm3, :][:] .* psi_s_n[d2, l, In][:] )
										)[1]
									# lnn, lnm, lmn, lmm
									D3E[d1, l, d2, n, d3, n] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( d2H_nn[m2, mm3, :][:] .* psi_s_n[d1, l, In][:] )
										)[1]
									D3E[d1, l, d2, n, d3, m] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( - d2H_nn[mm2, m3, :][:] .* psi_s_n[d1, l, In][:] )
										)[1]
									D3E[d1, l, d2, m, d3, n] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( - d2H_nn[m2, mm3, :][:] .* psi_s_n[d1, l, In][:] )
										)[1]
									D3E[d1, l, d2, m, d3, mm] +=  feps3[s] * (
 										2.0 * C[In, s]' * ( d2H_nn[m2, mm3, :][:] .* psi_s_n[d1, l, In][:] )
										)[1]
								end 	# loop for atom l

								# 8 parts:  H_{nn,nnn}, H_{nn,nnm}, H_{nn,nmn}, H_{nn,nm'm''}
								# 			H_{nn,mm'm''}, H_{nn,mmn}, H_{nn,mnm}, H_{nn,mnn}
								# a third loop for neighbours
								for i_l = 1:length(neigs)
									ll = neigs[i_l]
			    			    	Ill = indexblock(ll, tbm)
									ll1 = 3*(i_l-1) + d1
									ll2 = 3*(i_l-1) + d2
									ll3 = 3*(i_l-1) + d3

									# nnn, nnm, nmn, nmm
									D3E[d1, n, d2, n, d3, n] +=  feps3[s] * (
									 	C[In, s]' * ( - d3H_nn[m1, mm2, ll3, :][:] .* C[In, s] )
										)[1]
									D3E[d1, n, d2, n, d3, m] +=  feps3[s] * (
									 	C[In, s]' * ( d3H_nn[mm1, ll2, m3, :][:] .* C[In, s] )
										)[1]
									D3E[d1, n, d2, m, d3, n] +=  feps3[s] * (
									 	C[In, s]' * ( d3H_nn[mm1, m2, ll3, :][:] .* C[In, s] )
										)[1]
									D3E[d1, n, d2, m, d3, mm] +=  feps3[s] * (
									 	C[In, s]' * ( - d3H_nn[ll1, m2, mm3, :][:] .* C[In, s] )
										)[1]
									# mnn, mnm, mmn, mmm
									D3E[d1, m, d2, n, d3, n] +=  feps3[s] * (
									 	C[In, s]' * ( d3H_nn[m1, mm2, ll3, :][:] .* C[In, s] )
										)[1]
									D3E[d1, m, d2, n, d3, mm] +=  feps3[s] * (
									 	C[In, s]' * ( - d3H_nn[m1, ll2, mm3, :][:] .* C[In, s] )
										)[1]
									D3E[d1, m, d2, mm, d3, n] +=  feps3[s] * (
									 	C[In, s]' * ( - d3H_nn[m1, mm2, ll3, :][:] .* C[In, s] )
										)[1]
									D3E[d1, m, d2, mm, d3, ll] +=  feps3[s] * (
									 	C[In, s]' * ( d3H_nn[m1, mm2, ll3, :][:] .* C[In, s] )
										)[1]

								end 	# loop for neighbours i_l
							end		# loop for neighbours i_m

						end		# loop for d3
					end		# loop for d2
				end		# loop for d1

			end		# loop for neighbours i_n
    	end		# loop for atomic sites
    end		# loop for eigenstates

    return D3E
end










##################################################
### MODELS




include("NRLTB.jl")
include("tbtoymodel.jl")

end
