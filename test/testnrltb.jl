
TB=TightBinding
at = (1,2,2) * bulk("Si", pbc=false, cubic=true)
@show length(at)
β, eF, fixed_eF = 1.0, 0.0, true
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(β, eF, fixed_eF),
                           bzquad = TB.GammaPoint() )
                           # bzquad=TB.MPGrid(at, (10,0,0)) )
# tbm.smearing.fixed_eF = false
# tbm.smearing = TB.FermiDiracSmearing(100.0)
# TB.update!(at, tbm)
# tbm.fixed_eF = true

TB.update!(at, tbm)

# @time forces(tbm, at)
# @time forces(tbm, at)
# @time forces(tbm, at)
#
# @time skhg = TB.SparseSKHgrad(tbm.H, at)
# @time skhg = TB.SparseSKHgrad(tbm.H, at)
# @time skhg = TB.SparseSKHgrad(tbm.H, at)

# @time TB.forcesnew(tbm, at)
# @time TB.forcesnew(tbm, at)
# @time TB.forcesnew(tbm, at)


Fold = TB.forces(tbm, at);
Fnew = TB.forcesnew(tbm, at);

@show maximum(norm.(Fold - Fnew))

# TB.forcesnew(tbm, at)
# @profile TB.forcesnew(tbm, at)
# Profile.print()

# TB.SparseSKHgrad(tbm.H, at)
# @profile TB.forcesnew(tbm, at)
# Profile.print()


# TB.update!(at, tbm)
# skhg = TB.SparseSKHgrad(tbm.H, at)
# # @time TB._forcesnew_k(at, tbm, tbm.H, JVecF(rand(3)), skhg)
# k = tbm.bzquad.k[1]
# TB._forcesnew_k(at, tbm, tbm.H, k, skhg)
# @profile  TB._forcesnew_k(at, tbm, tbm.H, k, skhg)
# Profile.print()

quit()


X = copy(positions(at)) |> mat
X[:, 2] += [0.123; 0.234; 0.01]
set_positions!(at, vecs(X))

println("-------------------------------------------")
println("Testing NRLTBModel: ")
@show length(at)
print("check that hamiltonian evaluates ... ")
H, M = hamiltonian(tbm, at)
println("ok.")
# print("check that `densitymatrix` evaluates ... ")
# Γ = densitymatrix(tbm, at)
# println("ok.")
print("check that `energy` evaluates ... ")
E = energy(tbm, at)
println("ok : E = ", E, ".")
print("check that `forces` evaluates ... ")
frc = forces(tbm, at) |> mat
println("ok : |f|∞ = ", vecnorm(frc, Inf), ".")

println("-------------------------------------------")
println("  Finite-difference test with NRLTBModel:  ")
println("-------------------------------------------")
fdtest(tbm, at, verbose=true)
println("-------------------------------------------")
