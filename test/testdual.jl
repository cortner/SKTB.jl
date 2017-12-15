using JuLIP, JuLIP.Potentials
using JuLIP.Potentials: site_energy, site_energy_d
using TightBinding
using TightBinding: FermiDiracSmearing
TB = TightBinding
NRLTB = TB.NRLTB

@testset "Dual Method Partial Energies" begin
println("Testing Dual Method Implementation of Partial Energies")

# test parameters
beta = 10.0        # temperature / smearing paramter: 10 to 50 for room temperature
n0 = 5             # site index where we compute the site energy
DIM = (2,3,4)

# define the model
tbm = NRLTB.NRLTBModel(:Si, FermiDiracSmearing(beta), bzquad = TB.GammaPoint() )
at = bulk("Si", pbc = (false, false, true), cubic=false) * DIM
@show length(at)

# TB.set_δNel!(tbm, at)
E = energy(tbm, at)
∑En = sum( site_energy(tbm, at, n) for n = 1:length(at) )
println("Testing that the decompositions site energy sums to total energy")
@show E - ∑En
@test E - ∑En ≈ 0.0
JuLIP.rattle!(at, 0.02)
X = positions(at) |> mat


println("Finite difference test for site energy derivatives")
Es  = site_energy(tbm, at, n0)
dEs = site_energy_d(tbm, at, n0) |> mat
errors = Float64[]
for p = 2:9
   h = 0.1^p
   dEsh = zeros(dEs)
   for n = 1:length(X)
      X[n] += h
      set_positions!(at, X)
      Esh = site_energy(tbm, at, n0)
      dEsh[n] = (Esh - Es) / h
      X[n] -= h
   end
   push!(errors, vecnorm(dEs - dEsh, Inf))
   println( " ", p, " | ", errors[end])
end
@test minimum(errors) < 1e-3 * maximum(errors)


println("Finite difference test for partial_energy")
Idom = [3, 4, 5, 6]
Es  = partial_energy(tbm, at, Idom)
dEs = partial_energy_d(tbm, at, Idom) |> mat
errors = Float64[]
for p = 2:9
   h = 0.1^p
   dEsh = zeros(dEs)
   for n = 1:length(X)
      X[n] += h
      set_positions!(at, X)
      Esh = partial_energy(tbm, at, Idom)
      dEsh[n] = (Esh - Es) / h
      X[n] -= h
   end
   push!(errors, vecnorm(dEs - dEsh, Inf))
   println( " ", p, " | ", errors[end])
end
@test minimum(errors) < 1e-3 * maximum(errors)


end
