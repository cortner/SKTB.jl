
# # TODO: where does this go?????
# # TODO: default evaluate!; should this potentially go into JuLIP.Potentials?
# evaluate!(pot, r, R, target)  = copy!(target, evaluate(pot, r, R))
# evaluate_d!(pot, r, R, target)  = copy!(target, evaluate_d(pot, r, R))
# grad(pot, r, R) = R .* (evaluate_d(pot, r, R) ./ r)'
# grad!(p, r, R, G) = copy!(G, grad(p, r, R))




# """
# uses spectral decomposition to compute Emin, Emax, eF
# for the configuration `at` and stores it in `calc`
# """
# function calibrate2!(calc::PEXSI, at::AbstractAtoms,
#                      beta::Float64; nkpoints=(4,4,4), eF = :auto )
#    tbm = calc.tbm
#    tbm.potential = FermiDiracSmearing(beta)
#    tbm.fixed_eF = false
#    tbm.eF = 0.0
#    tbm.nkpoints, nkpoints_old = nkpoints, tbm.nkpoints
#    # this computes the spectrum and fermi-level
#    H, M = hamiltonian(calc.tbm, at)
#    ϵ = eigvals(full(H), full(M))
#    if eF == :auto
#       tbm.eF = 0.5 * sum(extrema(ϵ))
#    else
#       tbm.eF = eF
#    end
#    tbm.potential.eF = tbm.eF
#    tbm.fixed_eF = true
#    calc.Emin = 0.0
#    calc.Emax = maximum( abs(ϵ - tbm.eF) )
#    return calc
# end




# TB = TightBinding


# # 2D geometry: cross-over is at 0.05% sparsity!!!
# for N = 6:2:20
#    at = (N,N,1) * bulk("Al", pbc=false, cubic=true)
#    tbm = TightBinding.ToyTBModel(r0=2.5, rcut=5.5)
#    H, M = hamiltonian(tbm, at)
#    Hs = sparse(H)
#    println("N = $N | sparsity: $(nnz(Hs)/length(Hs)) %")
#    println("  full LU:")
#    @time lufact(H.data)
#    @time lufact(H.data)
#    @time lufact(H.data)
#    println("  sparse LU:")
#    @time lufact(Hs)
#    @time lufact(Hs)
#    @time lufact(Hs)
# end


# # 3D: cross-over is at 0.03% sparsity!!!
# for N = 3:8
#    at = (N,N,N) * bulk("Al", pbc=false, cubic=true)
#    tbm = TightBinding.ToyTBModel(r0=2.5, rcut=5.5)
#    H, M = hamiltonian(tbm, at)
#    Hs = sparse(H)
#    println("N = $N | sparsity: $(nnz(Hs)/length(Hs)) %")
#    println("  full LU:")
#    @time lufact(H.data)
#    @time lufact(H.data)
#    @time lufact(H.data)
#    println("  sparse LU:")
#    @time lufact(Hs)
#    @time lufact(Hs)
#    @time lufact(Hs)
# end


# 2D geometry NRL: cross-over is at 0.06% sparsity!!!
# 3D, NRL: ca. 0.06 as well
for N = 2:6
   at = (N,N,N) * bulk("Si", pbc=false, cubic=true)
   tbm = TightBinding.NRLTB.NRLTBModel(:Si, FermiDiracSmearing(1.0))
   H, M = hamiltonian(tbm, at)
   Hs = sparse(H)
   println("N = $N | sparsity: $(nnz(Hs)/length(Hs)) %")
   println("  full LU:")
   @time lufact(H.data)
   @time lufact(H.data)
   @time lufact(H.data)
   println("  sparse LU:")
   @time lufact(Hs)
   @time lufact(Hs)
   @time lufact(Hs)
end







# import Base.-
# -(A::AbstractVector{JVecF}, a::JVecF) = JVecF[v - a for v in A]
#
# dott(a::JVecF, A::AbstractVector{JVecF}) = JVecF[dot(a, v) for v in A]
#
#
# # # append to triplet format: version 1 for H and M (non-orth TB)
# function _append!{NORB}(H::SKHamiltonian{NONORTHOGONAL, NORB},
#                   It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, idx)
#    @inbounds for i = 1:NORB, j = 1:NORB
#       idx += 1
#       It[idx] = In[i]
#       Jt[idx] = Im[j]
#       Ht[idx] = H_nm[i,j] * exp_i_kR
#       Mt[idx] = M_nm[i,j] * exp_i_kR
#    end
#    return idx
# end
#
# # append to triplet format: version 2 for H only (orthogonal TB)
# function _append!{NORB}(H::SKHamiltonian{ORTHOGONAL, NORB},
#                   It, Jt, Ht, _Mt_, In, Im, H_nm, _Mnm_, exp_i_kR, idx)
#    @inbounds for i = 1:NORB, j = 1:NORB
#       idx += 1
#       It[idx] = In[i]
#       Jt[idx] = Im[j]
#       Ht[idx] = H_nm[i,j] * exp_i_kR
#    end
#    return idx
# end
#
# # inner SKHamiltonian assembly
# #
# # NOTES:
# #  * exp_i_kR = complex multiplier needed for BZ integration
# #  * we could use cell * S instead of R[m] - (X[neigs[m]] - X[n])
# #       but this would actually be less efficient, and less clear to read
# #
# #
# function assemble!{ISORTH, NORB}(H::SKHamiltonian{ISORTH, NORB},
#                                  k, It, Jt, Ht, Mt, nlist, X)
#
#    # TODO: H_nm, M_nm, bonds could all be MArrays
#    idx = 0                     # initialise index into triplet format
#    H_nm = zeros(NORB, NORB)    # temporary arrays for computing H and M entries
#    M_nm = zeros(NORB, NORB)
#    bonds = zeros(nbonds(H))     # temporary array for storing the potentials
#
#    # loop through sites
#    for (n, neigs, r, R, _) in sites(nlist)
#       In = indexblock(n, H)     # index-block for atom index n
#       # loop through the neighbours of the current atom (i.e. the bonds)
#       for m = 1:length(neigs)
#          U = R[m]/r[m]
#          # hamiltonian block
#          sk!(H, U, hop!(H, r[m], bonds), H_nm)
#          # overlap block (only if the model is NONORTHOGONAL)
#          ISORTH || sk!(H, U, overlap!(H, r[m], bonds), M_nm)
#          # add new indices into the sparse matrix
#          Im = indexblock(neigs[m], H)
#          exp_i_kR = exp( im * dot(k, R[m] - (X[neigs[m]] - X[n])) )
#          idx = _append!(H, It, Jt, Ht, Mt, In, Im, H_nm, M_nm, exp_i_kR, idx)
#       end
#
#       # now compute the on-site blocks;
#       # TODO: revisit this (can one do the scalar temp trick again?) >>> only if we assume diagonal!
#       # on-site hamiltonian block
#       onsite!(H, r, R, H_nm)
#       # on-site overlap matrix block (only if the model is NONORTHOGONAL)
#       ISORTH || overlap!(H, M_nm)
#       # add into sparse matrix
#       idx = _append!(H, It, Jt, Ht, Mt, In, In, H_nm, M_nm, 1.0, idx)
#    end
#
#    # convert M, H into Sparse CCS and return
#    #   NOTE: The conversion to sparse format accounts for about 1/2 of the
#    #         total cost. Since It, Jt are in an ordered format, it should be
#    #         possible to write a specialised code that converts it to
#    #         CCS much faster, possibly with less additional allocation?
#    #         another option would be to store a single It, Jt somewhere
#    #         for ALL the hamiltonians, and store multiple Ht, Mt and convert
#    #         these "on-the-fly", depending on whether full or sparse is needed.
#    #         but at the moment, eigfact cost MUCH more than the assembly,
#    #         so we could choose to stop here.
#    return ISORTH ? (sparse(It, Jt, Ht), I) :
#                    (sparse(It, Jt, Ht), sparse(It, Jt, Mt))
# end
