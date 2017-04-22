
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




TB = TightBinding


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
