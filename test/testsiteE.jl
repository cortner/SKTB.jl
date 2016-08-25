
# test parameters
beta = 10.0        # temperature / smearing paramter
                     # 10 to 50 for room temperature
n0 = 1            # site index where we compute the site energy
NQUAD = (4, 6, 8, 10)     # number of contour points
DIM = (1,4,4)

TB=TightBinding

# define the model
tbm = TB.NRLTB.NRLTBModel(elem = TB.NRLTB.Si_sp, nkpoints = (0,0,0))
calc = TB.Contour.ContourCalculator(tbm, 0)

# use a mini-system to pre-compute the Fermi-level and energy bounds
at = Atoms("Si", pbc=(true,true,true))
TB.Contour.calibrate!(calc, at, beta, nkpoints=(6,6,6))

# now the real system to test on
at = DIM * Atoms("Si", pbc=(true,true,true), cubic=true)
JuLIP.rattle!(at, 0.02)

# compute the site energy the old way
Eold = TB.site_energy(tbm, at, n0)
println("Old Site Energy (via spectral decomposition)")
println(Eold)

println("Testing that the old site energies sum to total energy")
Etot = TB.energy(tbm, at)
Es = [TB.site_energy(tbm, at, n) for n = 1:length(at)]
@show Etot - sum(Es)
@assert abs(Etot - sum(Es)) < 1e-10

# now try the new one
println("Convergence of Contour integral implementation")
for nquad in NQUAD
   calc.nquad = nquad
   Enew = TB.Contour.site_energy(calc, at, n0)
   println("nquad = ", nquad, "; error = ", abs(Enew - Eold))
end
