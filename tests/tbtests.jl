
push!(LOAD_PATH, ".", "..")

using TightBinding

println("Testing FermiDiracSmearing")
fd = TightBinding.FermiDiracSmearing(1.0, 0.0)
test_ScalarFunction(fd, -0.5 + rand(10))


