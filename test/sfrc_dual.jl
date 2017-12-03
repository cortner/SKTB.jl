using JuLIP, JuLIP.Potentials
using JuLIP.Potentials: site_energy, site_energy_d
using TightBinding
TB = TightBinding

# test parameters
beta = 10.0        # temperature / smearing paramter: 10 to 50 for room temperature
n0 = 5             # site index where we compute the site energy
DIM = (2,3,4)

# define the model
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(beta), bzquad = TB.GammaPoint() )
# check that the site energies add up to the total
at = bulk("Si", pbc = (false, false, true), cubic=false) * DIM
@show length(at)

# TB.set_δNel!(tbm, at)
E = energy(tbm, at)
∑En = sum( site_energy(tbm, at, n) for n = 1:length(at) )
println("Testing that the decompositions site energy sums to total energy")
@show E - ∑En
JuLIP.rattle!(at, 0.02)
X = positions(at) |> mat

println("Finite difference test for energy derivatives")
E   = energy(tbm, at)
dE  = forces(tbm, at) |> mat
dEh = []
for p = 2:9
   h = 0.1^p
   dEh = zeros(dE)
   for n = 1:length(X)
      X[n] += h
      set_positions!(at, X)
      Eh = energy(tbm, at)
      dEh[n] = (Eh - E) / h
      X[n] -= h
   end
   println( " ", p, " | ", vecnorm(dE + dEh, Inf) )
end
set_positions!(at, X)

println("Finite difference test for site energy derivatives")
Es  = site_energy(tbm, at, n0)
dEs = site_energy_d(tbm, at, n0) |> mat
dEsh = []
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
   println( " ", p, " | ", vecnorm(dEs - dEsh, Inf) )
end
