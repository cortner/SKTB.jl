
TB=TightBinding
at = (1,2,2) * bulk("Si", pbc=false, cubic=true)
β, eF, fixed_eF = 1.0, 0.0, true
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(β, eF, fixed_eF),
                             bzquad=TB.MPGrid(at, (4,2,0)) )
# tbm.smearing.fixed_eF = false
# tbm.smearing = TB.FermiDiracSmearing(100.0)
# TB.update!(at, tbm)
# tbm.fixed_eF = true

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
