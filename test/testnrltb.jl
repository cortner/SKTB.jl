
TB=TightBinding

@testset "NRLTB" begin

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
   at = (1,2,2) * bulk("Si", pbc=(true, true, true), cubic=true)
   @show length(at)
   β, fixed_eF = 1.0, 0.0, true
   tbm = TB.NRLTB.NRLTBModel(:Si, orbitals = orbitals,
                             TB.FermiDiracSmearing(β, fixed_eF=fixed_eF),
                             # bzquad = TB.GammaPoint() )
                             bzquad=TB.MPGrid(at, (4,0,0)) )
   print("Test setting Nel ... ")
   TB.set_δNel!(tbm, at)
   println("ok.")
   @test true
   @show TB.get_eF(tbm.potential)

   # perturb positions a bit
   # X = positions(at)
   # X[2] += 0.1 * rand(JVecF)
   # set_positions!(at, X)

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
   println("ok : |f|∞ = ", vecnorm(frc, Inf), ".")
   @test true

   # println("-------------------------------------------")
   # println("  Finite-difference test with NRLTBModel:  ")
   # println("-------------------------------------------")
   # @test fdtest(tbm, at, verbose=true)
   # println("-------------------------------------------")
end

end
