
@testset "ToyModel" begin

println("Testing TB Toy Model")

TB = TightBinding
at = (1,2,3) * bulk("Al", pbc=false, cubic=true)
tbm = TightBinding.ToyTBModel(r0=2.5, rcut=4.9)

X = positions(at)
X[2] += [0.123, 0.234, 0.01]
set_positions!(at, X)
@show length(at)


println("-------------------------------------------")
println("Testing ToyTBModel: ")
@show length(at)
print("check that hamiltonian evaluates ... ")
H, M = hamiltonian(tbm, at)
println("ok.")
print("check that `densitymatrix` evaluates ... ")
Γ = densitymatrix(tbm, at)
println("ok.")@test true
print("check that `energy` evaluates ... ")
E = energy(tbm, at)
println("ok : E = ", E, ".")
print("check that `forces` evaluates ... ")
frc = forces(tbm, at)
println("ok : |f|∞ = ", maximum(norm.(frc)), ".")
@test true

println("-------------------------------------------")
println("  Finite-difference test with ToyTBModel:  ")
println("-------------------------------------------")
@test fdtest(tbm, at, verbose=true)
println("-------------------------------------------")

end
