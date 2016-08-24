
# test parameters
beta = 300.0      # temperature / smearing paramter
n0 = 1            # site index where we compute the site energy
NQUAD = (5, 10, 20)     # number of contour points



TB=TightBinding

# define the model
tbm = TB.NRLTB.NRLTBModel(elem = TB.NRLTB.Si_sp, nkpoints = (0,0,0))
calc = TB.Contour.ContourCalculator(tbm, 0)

# use a mini-system to pre-compute the Fermi-level and energy bounds
at = Atoms("Si", pbc=(true,true,true))
TB.Contour.calibrate!(calc, at, beta, nkpoints=(6,6,6))

# now go to the real system
at = (3,3,3) * Atoms("Si", pbc=(false,false,false), cubic=true)
JuLIP.rattle!(at, 0.02)

# compute the site energy the old way
Eold = TB.site_energy(tbm, at, n0)
println("Old Site Energy (via spectral decomposition)")
println(Eold)

# now try the new one
println("Contour integral implementation")
for nquad in NQUAD
   calc.nquad = nquad
   Enew = TB.Contour.site_energy(calc, at, n0)
   println(Enew, " <<< nquad = ", nquad)
end

# # timing test
# println("timing with nquad = ", calc.nquad, "  (ca. 6 digits)")
# @time TB.Contour.site_energy(calc, at, n0)
# @time TB.Contour.site_energy(calc, at, n0)
# @show length(at) * tbm.norbitals
