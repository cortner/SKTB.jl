
using JuLIP, JuLIP.Potentials, SKTB
using JuLIP.Potentials: site_energy, site_energy_d

TB=SKTB

println("Test Kwon TB Model")
at = (1,2,2) * bulk(:Si, pbc=(true, false, true), cubic=true)
@show length(at)
β, fixed_eF = 30.0, true
tbm = TB.Kwon.KwonTBModel(potential = FermiDiracSmearing(β),
                          # potential = TB.GrandPotential(β, 0.0),
                          # bzquad=TB.MPGrid(at, (4,0,0)) )
                          bzquad = TB.GammaPoint() )
print("Test setting Nel ... ")
TB.set_δNel!(tbm, at)
println("ok.")
@test true
@show TB.get_eF(tbm.potential)

# perturb positions a bit
X = positions(at)
X[2] += 0.1 * rand(JVecF)
set_positions!(at, X)

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
println("ok : |f|∞ = ", norm(frc, Inf), ".")
@test true

println("  Finite-difference test:  ")
@test fdtest(tbm, at, verbose=true)
println("  Finite-difference test without the repulsive part:  ")
Vrep, tbm.Vrep = tbm.Vrep, JuLIP.Potentials.ZeroSitePotential()
@test fdtest(tbm, at, verbose=true)
tbm.Vrep = Vrep

# ==================== site energy tests ==================

n0 = 1
NQUAD = (4, 8, 16, 20, 25, 30)

E = energy(tbm, at)
∑En = sum( site_energy(tbm, at, n) for n = 1:length(at) )
println("Testing that the spectral-decompositions site energy sums to total energy")
@show E - ∑En
@test abs(E - ∑En) < 1e-10


calc = TB.PEXSI(tbm, 5, [n0])
JuLIP.rattle!(at, 0.02)
# calibrate the PEXSI calculator on a mini-system
print("calibrating . . . ")
TB.calibrate!(calc, at; at_train = bulk(:Si, pbc=true), npoles = 8)
println("done."); @test true
# output some useful info if we are watching the tests...
@show length(at)
@show TB.get_EminEmax(at)

# compute the site energy the old way and compare against the PEXSI calculation
Eold = TB.site_energy(tbm, at, n0)
println("Old Site Energy (via spectral decomposition): ", Eold)
println("Testing convergence of PEXSI site energy")
Enew = 0.0
for nquad in NQUAD
   TB.set_npoles!(calc, nquad)
   Enew = site_energy(calc, at, n0)
   println("nquad = ", nquad, "; error = ", abs(Enew - Eold))
end
@show Enew, Eold
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

