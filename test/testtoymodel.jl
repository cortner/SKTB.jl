

at = (1,2,3) * bulk("Al", pbc=false, cubic=true)
tbm = TightBinding.ToyTBModel(r0=2.5, rcut=8.0)

X = copy(positions(at)) |> mat
X[:, 2] += [0.123; 0.234; 0.01]
set_positions!(at, vecs(X))
# @show length(at)

println("-------------------------------------------")
println("Testing ToyTBModel: ")
@show length(at)
print("check that hamiltonian evaluates ... ")
H, M = hamiltonian(tbm, at)
println("ok.")
print("check that `densitymatrix` evaluates ... ")
Γ = densitymatrix(tbm, at)
println("ok.")

# print("check that `energy` evaluates ... ")
# E = energy(tbm, at)
# println("ok : E = ", E, ".")
# print("check that `forces` evaluates ... ")
# frc = forces(tbm, at) |> mat
# println("ok : |f|∞ = ", vecnorm(frc, Inf), ".")

# println("-------------------------------------------")
# println("  Finite-difference test with ToyTBModel:  ")
# println("-------------------------------------------")
# fdtest(tbm, at, verbose=true)
# println("-------------------------------------------")
