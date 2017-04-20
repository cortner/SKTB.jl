
TB=TightBinding
at = (1,2,2) * bulk("Si", pbc=(true, false, false), cubic=true)
@show length(at)
β, eF, fixed_eF = 1.0, 0.0, true
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(β, eF, fixed_eF),
                           # bzquad = TB.GammaPoint() )
                           bzquad=TB.MPGrid(at, (6,0,0)) )
# tbm.smearing.fixed_eF = false
# tbm.smearing = TB.FermiDiracSmearing(100.0)
# TB.update!(at, tbm)
# tbm.fixed_eF = true

X = positions(at)
X[2] += 0.1 * rand(JVecF)
set_positions!(at, X)

println("-------------------------------------------")
println("Testing NRLTBModel: ")
print("check that hamiltonian evaluates ... ")
H, M = hamiltonian(tbm, at)
println("ok.")
print("check that `densitymatrix` evaluates ... ")
Γ = densitymatrix(tbm, at)
println("ok.")
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
