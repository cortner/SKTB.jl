using JuLIP, JuLIP.Potentials, TightBinding
using JuLIP.Potentials: site_energy, site_energy_d
TB = TightBinding

@testset "Site Energy" begin

# test parameters
beta = 10.0        # temperature / smearing paramter: 10 to 50 for room temperature
n0 = 1            # site index where we compute the site energy
NQUAD = (4, 6, 8, 10)     # number of contour points
DIM = (1,2,3)


# define the model
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(beta), bzquad = TB.GammaPoint() )

# check that the site energies add up to the total
at = bulk("Si", pbc = (true, false, false), cubic=true) * DIM
TB.set_δNel!(tbm, at)
E = energy(tbm, at)
∑En = sum( site_energy(tbm, at, n) for n = 1:length(at) )
println("Testing that the spectral-decompositions site energy sums to total energy")
@show E - ∑En
@test abs(E - ∑En) < 1e-12



calc = TB.PEXSI(tbm, 5, [1])

# use a mini-system to pre-compute the Fermi-level and energy bounds
# TODO: need to allow calibrating on a different geometry and BZQuadratureRule
# print("calibrating . . . ")
# at = bulk("Si", pbc=(true,true,true))
# TB.Contour.calibrate!(calc, at, beta, nkpoints=(6,6,6))
# println("done.")

# now the real system to test on
at = DIM * bulk("Si", pbc=(false,false,false), cubic=true)
print("calibrating . . . ")
TB.update!(calc, at)
println("done."); @test true
JuLIP.rattle!(at, 0.02)
@show length(at)

@show get_info(at, :EminEmax)

# compute the site energy the old way
Eold = TB.site_energy(tbm, at, n0)
println("Old Site Energy (via spectral decomposition): ", Eold)
println("Testing convergence of PEXSI site energy")
Enew = 0.0
for nquad in NQUAD
   TB.set_npoles!(calc, nquad)
   Enew = site_energy(calc, at, n0)
   println("nquad = ", nquad, "; error = ", abs(Enew - Eold))
end
@test abs(Enew - Eold) < 1e-7

println("Test consistency of site forces")
TB.set_npoles!(calc, 8)
X = positions(at) |> mat
Es = site_energy(calc, at, n0)
dEs = site_energy_d(calc, at, n0) |> mat
dEsh = []
errors = Float64[]
for p = 2:9
   h = 0.1^p
   dEsh = zeros(dEs)
   for n = 1:length(X)
      X[n] += h
      set_positions!(at, X)
      Esh = site_energy(calc, at, n0)
      dEsh[n] = (Esh - Es) / h
      X[n] -= h
   end
   println( " ", p, " | ", vecnorm(dEs-dEsh, Inf) )
   push!(errors, vecnorm(dEs-dEsh, Inf))
end
@test minimum(errors) < 1e-3 * maximum(errors)

println("Test consistency of ContourCalculator for multiple sites")
Is = unique(mod(rand(Int, length(at) ÷ 3), length(at)) + 1)
Eold = sum( site_energy(tbm, at, n0) for n0 in Is )
Enew = 0.0
for nquad in NQUAD
   calc.nquad = nquad
   Enew = TB.partial_energy(calc, at, Is)
   println("nquad = ", nquad, "; rel-error = ", abs(Enew - Eold) / abs(Eold))
end
@test  abs(Enew - Eold) / abs(Eold) < 1e-6


println("Test consistency of multiple site forces")
calc.nquad = 8
X = copy( positions(at) |> mat )
Is = unique(mod(rand(Int, length(at) ÷ 3), length(at)) + 1)
Es, dEs = TB.pexsi_partial_energy(calc, at, Is, true)
dEs = dEs |> mat
dEsh = []
errors = []
println(" p  |  error ")
for p = 2:9
   h = 0.1^p
   dEsh = zeros(dEs)
   for n = 1:length(X)
      X[n] += h
      set_positions!(at, X)
      Esh = TB.partial_energy(calc, at, Is)
      dEsh[n] = (Esh - Es) / h
      X[n] -= h
   end
   println( " ", p, " | ", vecnorm(dEs-dEsh, Inf) )
   push!(errors, vecnorm(dEs-dEsh, Inf))
end
@test minimum(errors) < 1e-3 * maximum(errors)


end
