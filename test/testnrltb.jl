
using SparseArrays: issparse

TB=SKTB

println("Testing NRLTBModel: ")

if TESTDEPTH == 0
   Orbitals = ()
elseif TESTDEPTH == 1
   Orbitals = (:sp,)
else
   Orbitals = (:sp, :spd)
end

for orbitals in Orbitals
   println("Test NRLTB with $(orbitals)")
   at = (1,2,2) * bulk(:Si, pbc=(true, true, true), cubic=true)
   @show length(at)
   β, fixed_eF = 10.0, true
   tbm = TB.NRLTB.NRLTBModel(:Si, TB.GrandPotential(β, 0.0),
                           #  TB.FermiDiracSmearing(β, fixed_eF=fixed_eF),
                           orbitals = orbitals,
                           #   bzquad=TB.MPGrid(at, (4,0,0)) )
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

   println("-------------------------------------------")
   println("  Finite-difference test with NRLTBModel:  ")
   println("-------------------------------------------")
   @test fdtest(tbm, at, verbose=true)
   println("-------------------------------------------")
end

println("Test sparse hamiltonian assembly")
for N = 2:6
   at = (N,N,N) * bulk(:Si, pbc=false, cubic=true)
   tbm = SKTB.NRLTB.NRLTBModel(:Si, FermiDiracSmearing(1.0))
   H, M = hamiltonian(tbm, at)
   Hf, Mf =  SKTB.evaluate(tbm.H, at, zeros(3), T = Array)
   @show typeof(H)
   @test issparse(H) == (N > 3)
   @show norm(Array(H) - Hf, Inf)
   @test norm(Array(H) - Hf, Inf) == 0.0
end

