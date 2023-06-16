using JuLIP, JuLIP.Potentials, SKTB, SparseArrays
using JuLIP: site_energy, site_energy_d, energy, forces
TB = SKTB


# test parameters
beta = 10.0        # temperature / smearing paramter: 10 to 50 for room temperature
n0 = 1            # site index where we compute the site energy
NQUAD = (4, 8, 16, 20)     # number of contour points
DIM = (1,2,3)


# define the model
tbm = TB.NRLTB.NRLTBModel(:Si, TB.FermiDiracSmearing(beta), bzquad = TB.GammaPoint() )

# check that the site energies add up to the total
at = bulk(:Si, pbc = (true, false, false), cubic=true) * DIM
TB.set_δNel!(tbm, at)
E = energy(tbm, at)
∑En = sum( site_energy(tbm, at, n) for n = 1:length(at) )
println("Testing that the spectral-decompositions site energy sums to total energy")
@show E - ∑En
@test abs.(E - ∑En) < 1e-10

# now the real system to test on
calc = TB.PEXSI(tbm, 5, [1])
at = DIM * bulk(:Si, pbc=(false,false,false), cubic=true)
JuLIP.rattle!(at, 0.02)

# calibrate the PEXSI calculator on a mini-system
print("calibrating . . . ")
TB.calibrate!(calc, at; at_train = bulk(:Si, pbc=true), npoles = 8)
println("done."); @test true

# output some useful info if we are watching the tests...
@show length(at)
@show TB.get_EminEmax(at)

# compute the site energy the old way and compare against the PEXSI calculation
Eold = site_energy(tbm, at, n0)
println("Old Site Energy (via spectral decomposition): ", Eold)
println("Testing convergence of PEXSI site energy")
Enew = 0.0
for nquad in NQUAD
   TB.set_npoles!(calc, nquad)
   Enew = site_energy(calc, at, n0)
   println("nquad = ", nquad, "; error = ", abs(Enew - Eold))
end
@test abs(Enew - Eold) < 1e-5

println("Test consistency of site forces")
TB.set_npoles!(calc, 8)
X = positions(at) |> mat
Es = site_energy(calc, at, n0)
dEs = site_energy_d(calc, at, n0) |> mat
dEsh = []
errors = Float64[]
for p = 2:9
   h = 0.1^p
   dEsh = zero(dEs)
   for n = 1:length(X)
      X[n] += h
      set_positions!(at, X)
      Esh = site_energy(calc, at, n0)
      dEsh[n] = (Esh - Es) / h
      X[n] -= h
   end
   println( " ", p, " | ", norm(dEs-dEsh, Inf) )
   push!(errors, norm(dEs-dEsh, Inf))
end
@test minimum(errors) < 1e-3 * maximum(errors)

println("Test consistency of ContourCalculator for multiple sites")
Is = unique(mod.(rand(Int, length(at) ÷ 3), length(at)) .+ 1)
Eold = sum( site_energy(tbm, at, n0) for n0 in Is )
Enew = 0.0
for nquad in NQUAD
   calc.nquad = nquad
   Enew = energy(calc, at; domain = Is)
   println("nquad = ", nquad, "; rel-error = ", abs(Enew - Eold) / abs(Eold))
end
@test  abs(Enew - Eold) / abs(Eold) < 1e-5

println("Test consistency of multiple site forces")
calc.nquad = 8
X = copy( positions(at) |> mat )
Is = unique(mod.(rand(Int, length(at) ÷ 3), length(at)) .+ 1)
Es, dEs = TB.pexsi_partial_energy(calc, at, Is, true)
dEs = dEs |> mat
dEsh = []
errors = []
println(" p  |  error ")
for p = 2:9
   h = 0.1^p
   dEsh = zero(dEs)
   for n = 1:length(X)
      X[n] += h
      set_positions!(at, X)
      Esh = energy(calc, at; domain = Is)
      dEsh[n] = (Esh - Es) / h
      X[n] -= h
   end
   println( " ", p, " | ", norm(dEs-dEsh, Inf) )
   push!(errors, norm(dEs-dEsh, Inf))
end
@test minimum(errors) < 1e-3 * maximum(errors)


