
TB=TightBinding
at = Atoms("Si", repeatcell=(1,2,2), pbc=(false,false,false), cubic=true)
tbm = TB.NRLTB.NRLTBModel(elem=TB.NRLTB.Si_sp, nkpoints = (4,2,0))

X = copy(positions(at)) |> mat
X[:, 2] += [0.123; 0.234; 0.01]
set_positions!(at, pts(X))

println("-------------------------------------------")
println("Testing NRLTBModel: ")
@show length(at)
print("check that hamiltonian evaluates ... ")
H, M = hamiltonian(tbm, at)
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
