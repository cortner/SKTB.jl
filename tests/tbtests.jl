
push!(LOAD_PATH, ".", "..")

using TestAtoms
using TightBinding
using Potentials
using ASE

# println("Testing FermiDiracSmearing")
# fd = TightBinding.FermiDiracSmearing(1.0, 0.0)
# test_ScalarFunction(fd, -0.5 + rand(10))

# println("Test hopping potential for ToyTBModel")
# tbm = TightBinding.ToyTB.ToyTBModel(r0=2.5, rcut=8.0)
# test_ScalarFunction(tbm.hop, 1.0 + 8 * rand(100))


at = bulk("Al") #; cubic=true)
at = repeat(at, (3, 3, 1))
# print(length(at))

set_pbc!(at, [false, false, false])
#set_pbc!(at, [true, true, true])

tbm = TightBinding.ToyTB.ToyTBModel(r0=2.5, rcut=8.0)
X = positions(at)
f = TightBinding.potential_energy(at, tbm)
df = TightBinding.forces(at, tbm)[:]
    println("-----------------------------")
    println("  p | error ")
    println("----|------------------------")
    for p = 2:15
        h = 0.1^p
        dfh = zeros(length(df))
        for n = 1:length(df)
            X[n] += h
            set_positions!(at, X)
            dfh[n] = (TightBinding.potential_energy(at, tbm) - f) / h
            X[n] -= h
        end
    #@printf(" %2d | %1.7e \n", p, norm(df - dfh, Inf))
    @printf(" %2d | %1.7e \n", p, norm(dfh + df, Inf))
    end
println("-----------------------------")

